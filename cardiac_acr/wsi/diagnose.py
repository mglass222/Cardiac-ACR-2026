#!/usr/bin/env python
# coding: utf-8

"""
Backend-agnostic whole-slide diagnosis pipeline.

Accepts a :class:`cardiac_acr.backends.BackendClassifier` and runs:

  1. Preprocessing — SVS → scaled PNG + filtered PNG per slide.
     Tile scoring runs in-memory per slide; no tile/patch PNGs are
     written to disk.
  2. Classification — 224×224 patches are streamed from the SVS via
     OpenSlide (``_StreamingPatchDataset``) and pushed through
     ``classifier.classify``, saving softmax probabilities to
     ``classifier.saved_database_dir``. The DataLoader workers apply
     the <50% tissue filter in-line; a sentinel + custom collate drops
     rejects before the GPU sees them.
  3. Threshold filtering — drop patches whose top probability is below
     ``config.PREDICTION_THRESHOLD``.
  4. Slide-level diagnosis — aggregate filtered patch labels + a
     dedicated 1R2 focus count into a single rejection grade.

Predictions-dict keys are synthetic patch filenames of the form
``NNN-tile-rR-cC-xX-yY.png`` matching ``get_coords_from_name`` so
downstream consumers (count_1r2, annotate_png, annotate_svs) work
unchanged.

Outputs land under ``classifier.saved_database_dir`` and
``classifier.slide_dx_dir``. PNG/SVS annotation writing is not yet
wired in (see DEVELOPMENT_LOG "V1 scope").

Usage:
    python -m cardiac_acr.wsi.diagnose --backend {uni,resnet}

The legacy disk-based pipeline (materializing ~5 GB of tile/patch
PNGs per slide to ``TILE_DIR`` + ``SPLIT_TILE_DIR``) is preserved on
the ``disk-mode`` branch — useful if you need to eyeball intermediate
PNGs during filter tuning. ``git checkout disk-mode`` to recover.
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cardiac_acr import config as cg
from cardiac_acr.backends import BackendClassifier, load_classifier
from cardiac_acr.preprocessing import filter as wsi_filter
from cardiac_acr.preprocessing.filter_patches import (
    apply_image_filters,
    tissue_percent,
)
from cardiac_acr.preprocessing import slide
from cardiac_acr.preprocessing.tiles import score_tiles
from cardiac_acr.utils import cardiac_utils as utils
from cardiac_acr.wsi import count_1r2


# Classification batch size at inference. Kept local (not on the
# BackendClassifier) because it is tuning for the diagnosis loop, not
# a property of the backend itself.
_CLASSIFY_BATCH = 64

# DataLoader workers for OpenSlide read + tissue filter + transform. 8
# saturates the 2070 SUPER during classify on this project; bump if a
# faster GPU is still CPU-bound.
_CLASSIFY_WORKERS = 8

# Minimum tissue percentage a patch must have to be classified.
_TISSUE_PCT_MIN = 50


class _StreamingPatchDataset(Dataset):
    """Streams 224×224 patches from the SVS via OpenSlide at level 0.
    No intermediate patch PNGs on disk.

    Constructed from a list of top-scoring tiles (from
    ``TileSummary.top_tiles()``); each tile yields a 5x5 grid of
    224×224 patches at level-0 coordinates. Synthetic keys match the
    ``-x<int>-y<int>`` regex that count_1r2 / annotate_png /
    annotate_svs parse from filenames.

    OpenSlide handles do not survive ``fork()``, so the slide is opened
    lazily inside ``__getitem__`` — each DataLoader worker gets its
    own handle on first read. Covers num_workers=0 naturally.
    """

    def __init__(self, slide_number, top_tiles, transform):
        self.slide_number = slide_number
        self.transform = transform
        self._svs_path = slide.get_training_slide_path(slide_number)
        self._slide = None  # opened lazily per worker

        patch_size = cg.PATCH_SIZE
        coords = []
        for t in top_tiles:
            tile_w = t.o_c_e - t.o_c_s
            tile_h = t.o_r_e - t.o_r_s
            y_steps = (tile_h + patch_size - 1) // patch_size
            x_steps = (tile_w + patch_size - 1) // patch_size
            for j in range(y_steps):
                for i in range(x_steps):
                    coords.append((
                        t.r, t.c,
                        t.o_c_s + i * patch_size,
                        t.o_r_s + j * patch_size,
                    ))
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self._slide is None:
            self._slide = slide.open_slide(self._svs_path)

        tile_r, tile_c, x, y = self.coords[idx]
        patch_size = cg.PATCH_SIZE
        # read_region always returns patch_size×patch_size; OpenSlide
        # zero-pads past the slide edge (→ black on RGB convert).
        region = self._slide.read_region((x, y), 0, (patch_size, patch_size))
        rgb = np.asarray(region.convert("RGB"))
        filtered = apply_image_filters(rgb)

        name = (
            f"{int(self.slide_number):03d}-tile-"
            f"r{tile_r}-c{tile_c}-x{x}-y{y}.png"
        )
        if tissue_percent(filtered) < _TISSUE_PCT_MIN:
            return name, None
        tensor = self.transform(Image.fromarray(rgb))
        return name, tensor


def _drop_empty_collate(batch):
    kept = [(p, t) for p, t in batch if t is not None]
    if not kept:
        return [], None
    paths = [p for p, _ in kept]
    tensors = torch.stack([t for _, t in kept], dim=0)
    return paths, tensors


def _ensure_dirs(classifier: BackendClassifier):
    for d in (
        classifier.saved_database_dir,
        classifier.slide_dx_dir,
        cg.BACKEND_DIR,
    ):
        os.makedirs(d, exist_ok=True)


def classify_patches(slide_number, classifier: BackendClassifier,
                     batch_size=_CLASSIFY_BATCH,
                     num_workers=_CLASSIFY_WORKERS):
    """Run every patch through ``classifier.classify``, save softmax probs.

    Tile scoring runs in-memory; 224×224 patches are streamed from the
    SVS via OpenSlide. The DataLoader workers apply the <50% tissue
    filter in-line; a sentinel + ``_drop_empty_collate`` drops rejects
    before the batch hits the GPU.
    """
    t0 = time.time()

    tile_summary = score_tiles(slide_number)
    top_tiles = tile_summary.top_tiles()
    dataset = _StreamingPatchDataset(
        slide_number, top_tiles, classifier.transform
    )
    total_candidates = len(dataset)

    device = classifier.device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        collate_fn=_drop_empty_collate,
    )

    predictions = {}
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"slide {slide_number} classify",
        unit="batch",
        dynamic_ncols=True,
        mininterval=0.5,
        position=1,
        leave=False,
    )
    for batch_paths, batch in progress:
        if batch is None:
            continue
        batch = batch.to(device, non_blocking=True)
        logits = classifier.classify(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        for path, prob in zip(batch_paths, probs):
            predictions[path] = prob

    out_path = os.path.join(
        classifier.saved_database_dir,
        f"model_predictions_dict_{slide_number}.pickle",
    )
    with open(out_path, "wb") as fh:
        pickle.dump(predictions, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  kept {len(predictions)}/{total_candidates} tissue patches "
          f"({time.time() - t0:.1f}s) -> {out_path}")


def threshold_predictions(slide_number, classifier: BackendClassifier,
                          threshold=None):
    """Drop any patch whose top probability is below ``threshold``."""
    if threshold is None:
        threshold = cg.PREDICTION_THRESHOLD

    in_path = os.path.join(
        classifier.saved_database_dir,
        f"model_predictions_dict_{slide_number}.pickle",
    )
    with open(in_path, "rb") as fh:
        predictions = pickle.load(fh)

    filtered = {
        k: v for k, v in predictions.items()
        if np.asarray(v).max() > threshold
    }
    print(f"  thresholded: kept {len(filtered)}/{len(predictions)} patches")

    out_path = os.path.join(
        classifier.saved_database_dir,
        f"model_predictions_dict_{slide_number}_filtered.pickle",
    )
    with open(out_path, "wb") as fh:
        pickle.dump(filtered, fh, protocol=pickle.HIGHEST_PROTOCOL)


def diagnose(slide_number, classifier: BackendClassifier):
    """Aggregate patch-level predictions into a slide-level rejection grade."""
    in_path = os.path.join(
        classifier.saved_database_dir,
        f"model_predictions_dict_{slide_number}_filtered.pickle",
    )
    with open(in_path, "rb") as fh:
        filtered_predictions = pickle.load(fh)

    classes = classifier.classes
    class_count = {name: 0 for name in classes}
    for value in filtered_predictions.values():
        idx = int(np.argmax(value))
        class_count[classes[idx]] += 1

    # 1R2 focus count comes from the dedicated segmentation pipeline,
    # not from the patch classifier.
    one_r_two_count = count_1r2.main(slide_number, classifier.saved_database_dir)
    class_count["1R2"] = one_r_two_count
    print(f"  class counts: {class_count}")

    one_r_one_a = class_count.get("1R1A", 0)
    if one_r_one_a == 0 and one_r_two_count == 0:
        dx = "0R"
    elif one_r_one_a > 0 and one_r_two_count == 0:
        dx = "1R1A"
    elif one_r_two_count > 0:
        dx = "1R2" if one_r_two_count < 2 else "2R"
    else:
        dx = "UNK"

    print(f"  slide {slide_number} dx = {dx}")

    dx_pickle = os.path.join(
        classifier.slide_dx_dir,
        f"slide_dx_dict_{int(cg.PREDICTION_THRESHOLD * 100)}_pct.pickle",
    )
    if os.path.exists(dx_pickle):
        with open(dx_pickle, "rb") as fh:
            slide_dx_dict = pickle.load(fh)
    else:
        slide_dx_dict = {}
    slide_dx_dict[slide_number] = dx
    with open(dx_pickle, "wb") as fh:
        pickle.dump(slide_dx_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return dx, class_count


def run(backend: str, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    classifier = load_classifier(backend, device=device,
                                 checkpoint_path=checkpoint_path)
    print(f"backend: {classifier.name} ({len(classifier.classes)} classes)")
    _ensure_dirs(classifier)

    slides_to_process = utils.get_test_slide_numbers()
    print("slides to process:", slides_to_process)

    t0 = time.time()
    # count_1r2 reads PNG_SLIDE_DIR, and score_tiles reads the filtered
    # PNG. Both are cheap — one PNG and one filtered PNG per slide.
    slide.multiprocess_training_slides_to_images(image_num_list=slides_to_process)
    wsi_filter.multiprocess_apply_filters_to_images(image_num_list=slides_to_process)
    print(f"preprocessing done in {time.time() - t0:.1f}s")

    slide_bar = tqdm(
        slides_to_process,
        desc="slides",
        unit="slide",
        dynamic_ncols=True,
        position=0,
    )
    for slide_number in slide_bar:
        slide_bar.set_description(f"slides (current: {slide_number})")
        slide_t0 = time.time()

        classify_patches(slide_number, classifier)
        threshold_predictions(slide_number, classifier)
        diagnose(slide_number, classifier)

        tqdm.write(f"slide {slide_number} done in {time.time() - slide_t0:.1f}s")

    slide_bar.close()
    print(f"\nTotal time: {time.time() - t0:.1f}s")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cardiac_acr.wsi.diagnose",
        description="Run end-to-end WSI diagnosis with the chosen backend.",
    )
    parser.add_argument("--backend", choices=("uni", "resnet"), required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to the backend checkpoint (defaults to the "
                             "backend's MODEL_DIR).")
    args = parser.parse_args(argv)
    run(args.backend, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
