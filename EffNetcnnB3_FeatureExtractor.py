"""
EffNetcnn.py
============
CSCI323 Modern Artificial Intelligence — Spring 2026
University of Wollongong Dubai (UOWD)

Author  : Teammate A
Dataset : HAM10000 (~9,958 dermoscopy images, 7 classes)
          Splits   : produced by images_cleaner.py → PipelineRunner.run()
          Metadata : pre-processed by data_cleaner.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Image feature extraction module using a SELECTIVELY FINE-TUNED
EfficientNetB3 backbone pretrained on ImageNet.

This file does exactly five things and nothing else:
    1. Apply on-the-fly training augmentation (proportional class
       balancing, unique-combination enforcement) for the EXPORT/
       extraction stage
    2. Apply deterministic resize + normalise for val / test
    3. Fine-tune EfficientNetB3 on HAM10000 by unfreezing feature
       blocks 6, 7 and 8 (0-indexed) plus a temporary classifier head,
       monitoring validation loss and saving the best checkpoint
    4. Reload the best checkpoint, strip the classifier head, retain
       Global Average Pooling → 1536-dim embedding per image
    5. Save feature matrices and aligned metadata so Teammate C's
       multimodal fusion network can consume them directly

THIS FILE MUST NEVER:
    - Perform final classification (the temporary head exists ONLY to
      provide a training signal for fine-tuning; it is discarded before
      feature export)
    - Modify the train/val/test split from images_cleaner.py
    - Leak augmented data into the val or test sets

NOTE ON FINE-TUNING (added for B3 integration):
    The backbone is no longer fully frozen. Early feature blocks (0–5)
    remain frozen to preserve generic ImageNet edge/texture detectors,
    while blocks 6–8 are fine-tuned to specialise on dermoscopy. The
    fine-tuned weights are saved to a checkpoint and reloaded before
    extraction, so the exported embeddings come from the fine-tuned
    backbone — NOT the original frozen one. Determinism of the exported
    features is preserved because extraction always runs the SAME fixed
    best-checkpoint weights in eval() mode with gradients disabled.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POSITION IN THE PROJECT PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    data_cleaner.py       → cleans raw HAM10000 metadata CSV
    images_cleaner.py     → validates images, lesion-level split,
                            OHE encodes localization
    EffNetcnn.py          → THIS FILE — extracts image embeddings
    Teammate_B_classifier → end-to-end image-only CNN baseline
    Teammate C            → multimodal fusion (embeddings + metadata)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUTS → consumed by Teammate C
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {output_dir}/train_features.npy   float32  (N_train_aug, 1536)
    {output_dir}/val_features.npy     float32  (N_val,       1536)
    {output_dir}/test_features.npy    float32  (N_test,      1536)
    {output_dir}/train_meta.csv       aligned metadata for train split
    {output_dir}/val_meta.csv         aligned metadata for val split
    {output_dir}/test_meta.csv        aligned metadata for test split

Row i of *_features.npy corresponds exactly to row i of *_meta.csv.
Teammate C merges on image_id for safety verification, then concatenates:
    image embedding  (1536-dim)
  + metadata vector (age, sex, 15 OHE loc columns = 17-dim)
  = 1553-dim input to the fusion classifier.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OOP STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DatasetManager       — wraps train/val/test DataFrames, builds
                           HAMDataset objects and DataLoaders
    AugmentationManager  — builds transform pipelines; proportional
                           class balancing with unique-combination
                           tracking; generates augmented metadata rows
    FeatureExtractor     — loads frozen EfficientNetB3 backbone,
                           runs inference, returns (N, 1536) embeddings
    FeatureStorageManager— saves / loads .npy feature matrices and
                           aligned metadata CSV files
    ExtractionPipeline   — orchestrates all four classes end-to-end

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from images_cleaner import PipelineRunner
    from EffNetcnn import ExtractionPipeline

    runner = PipelineRunner(
        metadata_path = "HAM10000_metadata_cleaned.csv",
        image_dirs    = ["HAM10000_images_part1", "HAM10000_images_part2"],
    )
    train_df, val_df, test_df = runner.run()

    pipeline = ExtractionPipeline(
        train_df   = train_df,
        val_df     = val_df,
        test_df    = test_df,
        output_dir = "features",
        random_seed = 42,
    )
    pipeline.run()
"""

# ─────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────
import hashlib
import math
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120})

# ─────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────

# EfficientNetB3 expects 300×300 input (Tan & Le, 2019 — compound scaling).
# Using the architecture's native resolution ensures the pre-trained
# spatial feature detectors fire at the scale they were optimised for.
EFFICIENTNET_INPUT_SIZE: int = 300

# EfficientNetB3 Global Average Pooling output = 1536 features.
# This is the width of the final MBConv stage before the classifier head.
EFFICIENTNET_B3_FEATURE_DIM: int = 1536

# ImageNet normalisation statistics (mean and std per RGB channel).
# Must match the values used during pre-training so that input activations
# fall in the regime the backbone's weights expect.
# Ref: Deng et al. (2009) — "ImageNet: A Large-Scale Hierarchical Image Database"
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

# HAM10000 class map — integer codes match data_cleaner.py label encoding
DX_LABEL_MAP: Dict[int, str] = {
    0: "akiec",   # Actinic keratoses / Bowen's disease
    1: "bcc",     # Basal cell carcinoma
    2: "bkl",     # Benign keratosis-like lesions
    3: "df",      # Dermatofibroma
    4: "mel",     # Melanoma
    5: "nv",      # Melanocytic nevi (dominant class ~67%)
    6: "vasc",    # Vascular lesions
}
NUM_CLASSES: int = 7


# ─────────────────────────────────────────────────────────────────────────
# FINE-TUNING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────
# Imported strategy from EffNetB3_finetune_classifier.py. These govern the
# fine-tuning phase ONLY; they do not affect the deterministic feature
# extraction that follows.
#
# Blocks 6, 7, 8 of EfficientNetB3.features (0-indexed) are unfrozen and
# trained; blocks 0–5 stay frozen to preserve generic ImageNet features.
FINETUNE_UNFREEZE_FROM: int = 6      # features[6:] are trainable
FINETUNE_EPOCHS:        int = 30     # max epochs (early stopping may cut short)
FINETUNE_PATIENCE:      int = 8      # early-stopping patience on val loss
FINETUNE_LR:            float = 1e-4 # small lr — fine-tuning, not from scratch
FINETUNE_HEAD_HIDDEN:   int = 512    # temporary classifier head width
FINETUNE_HEAD_DROPOUT:  float = 0.4  # dropout in the temporary head

# Best-checkpoint filename (saved into output_dir, reloaded before export).
CHECKPOINT_NAME: str = "best_effnet_finetune.pt"


# ─────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    """Print a visual section divider for readability in logs."""
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def _set_seed(seed: int = 42) -> None:
    """
    Seed all RNGs for full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _print_class_counts(df: pd.DataFrame, label: str) -> None:
    """Print per-class sample counts for a given DataFrame split."""
    print(f"\n  {label}:")
    for code, name in DX_LABEL_MAP.items():
        n = (df["dx"] == code).sum()
        print(f"    {code} ({name:8s}): {n:,}")


# ══════════════════════════════════════════════════════════════════════════
# CLASS 1 — AugmentationManager
# ══════════════════════════════════════════════════════════════════════════

class AugmentationManager:
    """
    Builds transform pipelines and generates an augmented training
    DataFrame with proportional class balancing.

    ─── WHY AUGMENT AT THE FEATURE EXTRACTION STAGE? ────────────────────
    HAM10000 has severe class imbalance: 'nv' holds ~6,705 images
    (~67%) while 'df' has only 115 and 'vasc' 142. Without correction,
    any downstream model will be biased toward the majority class.

    Augmentation is handled here — not in images_cleaner.py — because:
        • images_cleaner.py is a pure integrity / splitting module and
          must never create new image data
        • Augmentation is applied only to training images; val/test
          receive only resize + normalise, preventing any data leakage
        • Keeping augmentation in one place means Teammate B's classifier
          and this extractor use an identical pipeline, enabling a fair
          performance comparison

    ─── AUGMENTATION TECHNIQUES — INDIVIDUAL JUSTIFICATIONS ─────────────

    RandomHorizontalFlip / RandomVerticalFlip (p=0.5 each)

    RandomRotation ±180°

    RandomResizedCrop (scale 0.85–1.0)

    ColorJitter — brightness + contrast only (factor 0.8–1.2)

    """

    # ── Discrete parameter grids for collision-detectable uniqueness ───
    ROTATION_CHOICES:   List[int]   = list(range(0, 360, 10))   # 36 values
    ZOOM_CHOICES:       List[float] = [0.85, 0.88, 0.91, 0.94,
                                       0.97, 1.00]              # 6 values
    BRIGHTNESS_CHOICES: List[float] = [0.80, 0.87, 0.93, 1.00,
                                       1.07, 1.13, 1.20]        # 7 values
    CONTRAST_CHOICES:   List[float] = [0.80, 0.87, 0.93, 1.00,
                                       1.07, 1.13, 1.20]        # 7 values
    # Unique combinations = 36 × 2 × 2 × 6 × 7 × 7 = 42,336

    MAX_RESAMPLE_ATTEMPTS: int = 200
    AUGMENTATION_CAP:      int = 8_000   # no class grows beyond this
    MIN_AUGMENTED_SIZE:    int = 300     # smallest class gets at least this

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed   = random_seed
        # Tracks used parameter hashes per original image_id
        self._seen_hashes: Dict[str, set] = {}

    # ── Build transform pipelines ──────────────────────────────────────

    def build_train_transform(self) -> T.Compose:
        """
        Stochastic on-the-fly training augmentation pipeline.

        Returns
        -------
        T.Compose  Full training transform pipeline.
        """
        return T.Compose([
            # Resize to EfficientNetB3 native resolution first.
            # Pre-resizing before stochastic ops ensures the crop/zoom
            # transforms work at the correct scale.
            T.Resize((EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE)),

            # Random horizontal flip — no canonical orientation in dermoscopy.
            T.RandomHorizontalFlip(p=0.5),

            # Random vertical flip — same justification as horizontal.
            T.RandomVerticalFlip(p=0.5),

            # Full 360° rotation — dermoscopy images have no fixed 'up'.
            # fill=0 pads corner triangles with black; after normalisation
            # these become ~-2.1 in channel 0, clearly outside natural
            # image range, signalling padding pixels to the network.
            T.RandomRotation(degrees=180, fill=0),

            # Tight zoom (85–100% of image area) — simulates dermoscope
            # placement variation. Tight scale avoids losing lesion borders.
            T.RandomResizedCrop(
                size=EFFICIENTNET_INPUT_SIZE,
                scale=(0.85, 1.00),
                ratio=(1.0, 1.0),
                antialias=True,
            ),

            # Brightness + contrast jitter. Hue/saturation excluded —
            # see class docstring for clinical reasoning.
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.0, hue=0.0),

            # Convert PIL → float32 tensor in [0, 1]
            T.ToTensor(),

            # Normalise to ImageNet statistics — required because the
            # backbone was trained on ImageNet with these exact values.
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def build_eval_transform(self) -> T.Compose:
        """
        Deterministic evaluation pipeline — resize and normalise only.

        No augmentation is applied at val/test time so that metrics are
        stable across runs and reflect true generalisation performance.

        Returns
        -------
        T.Compose  Eval transform pipeline.
        """
        return T.Compose([
            T.Resize((EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ── Proportional balancing ─────────────────────────────────────────

    def compute_augmentation_targets(
        self,
        df: pd.DataFrame,
    ) -> Dict[int, int]:
        """
        Compute per-class augmentation targets using logarithmic scaling.

        Parameters
        ----------
        df : pd.DataFrame  Training DataFrame with 'dx' column.

        Returns
        -------
        Dict[int, int]  { class_code : target_sample_count }
        """
        counts    = df["dx"].value_counts().sort_index().to_dict()
        max_count = max(counts.values())

        print(f"\n  {'Class':<8} {'Current':>10} {'Target':>10}  {'Multiplier':>12}")
        print(f"  {'-'*8} {'-'*10} {'-'*10}  {'-'*12}")

        targets: Dict[int, int] = {}
        for cls, count in sorted(counts.items()):
            if count >= max_count:
                scale = 1.0
            else:
                scale = math.log(max_count + 1) / math.log(count + 1)

            raw    = math.ceil(count * scale)
            target = max(self.MIN_AUGMENTED_SIZE,
                         min(raw, self.AUGMENTATION_CAP))
            targets[cls] = target

            name = DX_LABEL_MAP.get(cls, str(cls))
            print(f"  {name:<8} {count:>10,} {target:>10,}  "
                  f"{target / count:>11.2f}×")

        return targets

    def generate_augmented_dataframe(
        self,
        df:      pd.DataFrame,
        targets: Dict[int, int],
    ) -> pd.DataFrame:
        """
        Expand train_df to augmentation targets by inserting new metadata
        rows for synthetic images. No pixel data is written to disk.

        Each augmented row inherits all metadata columns from its original
        row (lesion_id, dx, age, sex, loc_*) and receives:
            image_id  = "{original_image_id}_aug_{N:03d}"
            aug_params = parameter tuple for deterministic replay

        Original rows are preserved with aug_params = None (eval transform
        is applied to them during extraction, not augmentation).

        Parameters
        ----------
        df      : pd.DataFrame    Original training DataFrame.
        targets : Dict[int, int]  Output of compute_augmentation_targets().

        Returns
        -------
        pd.DataFrame  Expanded DataFrame with augmented rows appended.
        """
        _header("STEP 2 — Generating Augmented Metadata Rows")

        df        = df.copy()
        df["aug_params"] = None
        new_rows: List[dict] = []

        for cls, target in sorted(targets.items()):
            cls_df  = df[df["dx"] == cls]
            current = len(cls_df)
            needed  = target - current

            name = DX_LABEL_MAP.get(cls, str(cls))
            if needed <= 0:
                print(f"  {name:<8}: no augmentation needed "
                      f"(current {current} >= target {target})")
                continue

            print(f"  {name:<8}: generating {needed:,} rows "
                  f"({current} → {target})")

            originals = cls_df.to_dict("records")
            aug_counter: Dict[str, int] = {}

            for i in range(needed):
                orig   = originals[i % len(originals)]
                img_id = orig["image_id"]

                aug_counter[img_id] = aug_counter.get(img_id, 0) + 1
                params = self._sample_unique_params(img_id)

                new_row                = dict(orig)
                new_row["image_id"]    = f"{img_id}_aug_{aug_counter[img_id]:03d}"
                new_row["aug_params"]  = params
                new_rows.append(new_row)

        aug_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        print(f"\n  Original rows   : {len(df):,}")
        print(f"  Synthetic rows  : {len(new_rows):,}")
        print(f"  Augmented total : {len(aug_df):,}")
        return aug_df

    # ── Unique parameter sampling ──────────────────────────────────────

    def _sample_unique_params(self, image_id: str) -> Tuple:
        """
        Sample a parameter tuple that has not been used for this image.

        WHY discrete parameter grids?
            Continuous sampling (e.g. Uniform(0, 360)) makes exact
            collision probability zero — the uniqueness constraint would
            never trigger even for visually near-identical augmentations.
            Discrete grids bound the parameter space and make collision
            detection meaningful.

        Parameters
        ----------
        image_id : str  Original image_id (before any _aug_ suffix).

        Returns
        -------
        tuple : (rotation_deg, h_flip, v_flip, zoom,
                 brightness_factor, contrast_factor)
        """
        if image_id not in self._seen_hashes:
            self._seen_hashes[image_id] = set()

        seen = self._seen_hashes[image_id]

        for _ in range(self.MAX_RESAMPLE_ATTEMPTS):
            params = (
                random.choice(self.ROTATION_CHOICES),
                random.randint(0, 1),
                random.randint(0, 1),
                random.choice(self.ZOOM_CHOICES),
                random.choice(self.BRIGHTNESS_CHOICES),
                random.choice(self.CONTRAST_CHOICES),
            )
            h = hashlib.sha256(str(params).encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                return params

        print(f"  WARNING: parameter space nearly exhausted for {image_id}")
        return params  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════
# CLASS 2 — HAMDataset
# ══════════════════════════════════════════════════════════════════════════

class HAMDataset(Dataset):
    """
    PyTorch Dataset for HAM10000 with on-the-fly transform application.

    """

    def __init__(
        self,
        df:              pd.DataFrame,
        train_transform: T.Compose,
        eval_transform:  T.Compose,
        is_train:        bool = True,
        force_train_transform: bool = False,
    ) -> None:
 
        self.df              = df.reset_index(drop=True)
        self.train_transform = train_transform
        self.eval_transform  = eval_transform
        self.is_train        = is_train
        self.force_train_transform = force_train_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Load image, apply transform, return (tensor, label, image_id).

        Returns
        -------
        tensor   : torch.Tensor  shape (3, 300, 300)
        label    : int           dx class code (0–6)
        image_id : str           for alignment with metadata CSV
        """
        row        = self.df.iloc[idx]
        image      = Image.open(row["filepath"]).convert("RGB")
        label      = int(row["dx"])
        image_id   = str(row["image_id"])
        aug_params = row.get("aug_params", None)

        if self.force_train_transform:
            # Fine-tuning phase: apply the stochastic training pipeline to
            # every sample (fresh augmentation each epoch on the raw split).
            tensor = self.train_transform(image)
        elif self.is_train and aug_params is not None:
            tensor = self._apply_aug_params(image, aug_params)
        else:
            tensor = self.eval_transform(image)

        return tensor, label, image_id

    @staticmethod
    def _apply_aug_params(
        image:      Image.Image,
        aug_params: Tuple,
    ) -> torch.Tensor:
        """
        Apply a deterministic augmentation from a stored parameter tuple.

        Uses torchvision.transforms.functional (TF) — explicit calls with
        fixed parameters — instead of stochastic T.* transforms so that
        the same parameter tuple always produces the exact same output.

        Parameters
        ----------
        image      : PIL Image (RGB)
        aug_params : (rotation_deg, h_flip, v_flip, zoom,
                      brightness_factor, contrast_factor)

        Returns
        -------
        torch.Tensor  shape (3, 300, 300), normalised
        """
        rot, h_flip, v_flip, zoom, brightness_f, contrast_f = aug_params

        # 1. Resize to target resolution
        img = TF.resize(image,
                        [EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE],
                        antialias=True)

        # 2. Rotation
        img = TF.rotate(img, angle=rot, fill=0)

        # 3. Horizontal flip
        if h_flip:
            img = TF.hflip(img)

        # 4. Vertical flip
        if v_flip:
            img = TF.vflip(img)

        # 5. Zoom via centre crop + resize back
        w, h   = img.size
        crop_w = int(w * zoom)
        crop_h = int(h * zoom)
        left   = (w - crop_w) // 2
        top    = (h - crop_h) // 2
        img    = TF.crop(img, top, left, crop_h, crop_w)
        img    = TF.resize(img,
                           [EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE],
                           antialias=True)

        # 6. Brightness adjustment
        img = TF.adjust_brightness(img, brightness_factor=brightness_f)

        # 7. Contrast adjustment
        img = TF.adjust_contrast(img, contrast_factor=contrast_f)

        # 8. To tensor → [0, 1] float32
        tensor = TF.to_tensor(img)

        # 9. ImageNet normalisation
        tensor = TF.normalize(tensor,
                              mean=list(IMAGENET_MEAN),
                              std=list(IMAGENET_STD))
        return tensor


# ══════════════════════════════════════════════════════════════════════════
# CLASS 3 — DatasetManager
# ══════════════════════════════════════════════════════════════════════════

class DatasetManager:
    """
    Wraps the three split DataFrames and builds HAMDataset + DataLoader
    objects for each split.

    Keeping DataLoader construction in its own class means
    ExtractionPipeline never needs to know about batch sizes, worker
    counts, or pin_memory — it just calls DatasetManager.build_loaders().
    """

    def __init__(
        self,
        augmented_train_df: pd.DataFrame,
        val_df:             pd.DataFrame,
        test_df:            pd.DataFrame,
        train_transform:    T.Compose,
        eval_transform:     T.Compose,
        batch_size:         int = 32,
        num_workers:        int = 2,
    ) -> None:
        self.augmented_train_df = augmented_train_df
        self.val_df             = val_df
        self.test_df            = test_df
        self.train_transform    = train_transform
        self.eval_transform     = eval_transform
        self.batch_size         = batch_size
        self.num_workers        = num_workers

    def build_loaders(
        self,
        device: torch.device,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build and return (train_loader, val_loader, test_loader).

        shuffle=False for all loaders because:
            - We need row order preserved to align features with metadata
            - The augmentation diversity comes from the aug_params column,
              not from DataLoader shuffling

        Parameters
        ----------
        device : torch.device  Used to set pin_memory for GPU speedup.

        Returns
        -------
        train_loader, val_loader, test_loader : DataLoader
        """
        pin = device.type == "cuda"

        train_ds = HAMDataset(
            self.augmented_train_df,
            self.train_transform,
            self.eval_transform,
            is_train=True,
        )
        val_ds = HAMDataset(
            self.val_df,
            self.train_transform,
            self.eval_transform,
            is_train=False,
        )
        test_ds = HAMDataset(
            self.test_df,
            self.train_transform,
            self.eval_transform,
            is_train=False,
        )

        def _make_loader(ds: Dataset) -> DataLoader:
            return DataLoader(
                ds,
                batch_size  = self.batch_size,
                shuffle     = False,
                num_workers = self.num_workers,
                pin_memory  = pin,
            )

        train_loader = _make_loader(train_ds)
        val_loader   = _make_loader(val_ds)
        test_loader  = _make_loader(test_ds)

        print(f"\n  DataLoaders built:")
        print(f"    Train : {len(train_ds):,} samples  "
              f"({len(train_loader)} batches)")
        print(f"    Val   : {len(val_ds):,} samples  "
              f"({len(val_loader)} batches)")
        print(f"    Test  : {len(test_ds):,} samples  "
              f"({len(test_loader)} batches)")

        return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════
# CLASS 4 — FineTunedEffNetB3  (model used for the fine-tuning phase)
# ══════════════════════════════════════════════════════════════════════════

class FineTunedEffNetB3(nn.Module):
    """
    EfficientNetB3 with partial backbone unfreeze + a temporary
    classification head, used ONLY to fine-tune blocks 6–8 on HAM10000.

    Strategy imported verbatim from EffNetB3_finetune_classifier.py:
        Frozen   : features[0:6]  — low-level ImageNet features
        Unfrozen : features[6:]   — features[6], [7], [8] fine-tuned
        Head     : GAP → FC(1536, 512) → ReLU → Dropout(0.4) → FC(512, 7)

    ─── ROLE IN THIS FILE ───────────────────────────────────────────────
    The classifier head provides the gradient signal that fine-tunes the
    unfrozen blocks. After training, the head is DISCARDED: feature
    extraction uses only `self.features → GAP → flatten`, yielding the
    same 1536-dim embedding contract as the original frozen extractor.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        base = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )

        # Freeze the entire backbone first.
        for param in base.parameters():
            param.requires_grad = False

        # Unfreeze the last three feature blocks: features[6], [7], [8].
        for param in base.features[FINETUNE_UNFREEZE_FROM:].parameters():
            param.requires_grad = True

        # Retain the backbone's feature extractor and pooling so that, after
        # fine-tuning, we can reuse exactly these submodules for extraction.
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.flatten  = nn.Flatten(start_dim=1)

        # Temporary classification head — discarded before feature export.
        self.classifier = nn.Sequential(
            nn.Linear(EFFICIENTNET_B3_FEATURE_DIM, FINETUNE_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(FINETUNE_HEAD_DROPOUT),
            nn.Linear(FINETUNE_HEAD_HIDDEN, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass (backbone + head) — used during training."""
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embedding-only forward pass (backbone, NO head) - used during
        feature extraction. Produces the (batch, 1536) representation.
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x


# ══════════════════════════════════════════════════════════════════════════
# CLASS 5 — FineTuner  (training loop with best-checkpoint logic)
# ══════════════════════════════════════════════════════════════════════════

class FineTuner:
    """
    Fine-tunes a FineTunedEffNetB3 on the raw training split and saves the
    best-validation-loss checkpoint.

    Training strategy imported from EffNetB3_finetune_classifier.py:
        Loss       : CrossEntropyLoss (plain — data balancing for the
                     export stage is handled separately by augmentation,
                     and XGBoost handles class decisions downstream)
        Optimizer  : Adam over trainable params only, lr=1e-4
        Scheduler  : ReduceLROnPlateau(mode="min", factor=0.5, patience=3)
        Stopping   : monitor val loss; save on improvement; early stop
                     after FINETUNE_PATIENCE epochs without improvement
    """

    def __init__(
        self,
        train_df:        pd.DataFrame,
        val_df:          pd.DataFrame,
        train_transform: T.Compose,
        eval_transform:  T.Compose,
        output_dir:      str,
        device:          torch.device,
        batch_size:      int = 32,
        num_workers:     int = 12,
    ) -> None:
        self.train_df        = train_df
        self.val_df          = val_df
        self.train_transform = train_transform
        self.eval_transform  = eval_transform
        self.output_dir      = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device          = device
        self.batch_size      = batch_size
        self.num_workers     = num_workers
        self.ckpt_path       = self.output_dir / CHECKPOINT_NAME

    # ── Per-epoch helpers (imported from the reference classifier) ──────

    @staticmethod
    def _train_one_epoch(model, loader, optimizer, criterion, device):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels, _ in loader:        # HAMDataset yields (img, label, image_id)
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)
        return total_loss / len(loader), correct / total

    @staticmethod
    def _evaluate_loader(model, loader, criterion, device):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels, _ in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
                total_loss += loss.item()
                correct    += (outputs.argmax(1) == labels).sum().item()
                total      += labels.size(0)
        return total_loss / len(loader), correct / total

    # ── Main training routine ───────────────────────────────────────────

    def fine_tune(self) -> "FineTunedEffNetB3":
        """
        Run the fine-tuning loop and return the model with the BEST
        validation-loss weights already reloaded.

        Returns
        -------
        FineTunedEffNetB3  Model on self.device, best checkpoint loaded,
                           ready for feature extraction.
        """
        _header("STEP 3 — Fine-tuning EfficientNetB3 (blocks 6–8)")

        # ── Build training / validation loaders ────────────────────────
        # Training uses the STOCHASTIC transform on the raw split, so we
        # mark is_train=True but pass no aug_params column → HAMDataset
        # applies self.train_transform on every sample (see below).
        pin = self.device.type == "cuda"

        train_ds = HAMDataset(
            self.train_df,
            self.train_transform,
            self.eval_transform,
            is_train=True,
            force_train_transform=True,   # stochastic per-epoch augmentation
        )
        val_ds = HAMDataset(
            self.val_df,
            self.train_transform,
            self.eval_transform,
            is_train=False,
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=pin,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=pin,
        )

        print(f"  Train samples : {len(train_ds):,}  "
              f"({len(train_loader)} batches, shuffle=True, stochastic aug)")
        print(f"  Val samples   : {len(val_ds):,}  "
              f"({len(val_loader)} batches, deterministic)")

        # ── Build model ────────────────────────────────────────────────
        model = FineTunedEffNetB3(num_classes=NUM_CLASSES).to(self.device)
        frozen    = sum(p.numel() for p in model.parameters()
                        if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        print(f"\n  Frozen params    : {frozen:,}")
        print(f"  Trainable params : {trainable:,}")
        print(f"  Unfrozen blocks  : features[6], features[7], features[8] "
              f"+ temporary classifier head")

        # ── Loss / optimiser / scheduler ───────────────────────────────
        # Plain CrossEntropyLoss (no class weights): the export-stage
        # augmentation already addresses imbalance for downstream fusion,
        # and XGBoost makes the final class decision.
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=FINETUNE_LR,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        # ── Training loop with best-checkpoint + early stopping ────────
        best_val_loss = float("inf")
        patience_ctr  = 0
        history = {"train_loss": [], "val_loss": [],
                   "train_acc": [],  "val_acc": []}

        for epoch in range(1, FINETUNE_EPOCHS + 1):
            tr_loss, tr_acc = self._train_one_epoch(
                model, train_loader, optimizer, criterion, self.device
            )
            va_loss, va_acc = self._evaluate_loader(
                model, val_loader, criterion, self.device
            )
            scheduler.step(va_loss)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:02d}/{FINETUNE_EPOCHS}  "
                  f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  |  "
                  f"Val Loss: {va_loss:.4f}  Acc: {va_acc:.4f}  |  "
                  f"LR: {lr_now:.2e}", flush=True)

            # Monitor validation loss; save whenever it improves.
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                patience_ctr  = 0
                torch.save(model.state_dict(), self.ckpt_path)
                print(f"    ✓ Val loss improved → saved {self.ckpt_path.name}")
            else:
                patience_ctr += 1
                if patience_ctr >= FINETUNE_PATIENCE:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(no val-loss improvement for "
                          f"{FINETUNE_PATIENCE} epochs)")
                    break

        # ── Learning curves (diagnostic) ───────────────────────────────
        self._plot_learning_curves(history)

        # ── Reload BEST checkpoint (NOT final-epoch weights) ───────────
        print(f"\n  Reloading best checkpoint (val loss={best_val_loss:.4f}) "
              f"before feature extraction...")
        model.load_state_dict(
            torch.load(self.ckpt_path, map_location=self.device)
        )
        print(f"  ✓ Best checkpoint loaded: {self.ckpt_path}")

        return model

    def _plot_learning_curves(self, history: Dict[str, List[float]]) -> None:
        """Save train/val loss and accuracy curves for the fine-tune run."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history["train_loss"], label="Train")
        ax1.plot(history["val_loss"],   label="Val")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
        ax2.plot(history["train_acc"], label="Train")
        ax2.plot(history["val_acc"],   label="Val")
        ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
        plt.suptitle("EfficientNetB3 Fine-tuning History (blocks 6–8)",
                     fontweight="bold")
        plt.tight_layout()
        out = self.output_dir / "finetune_learning_curves.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Fine-tune learning curves → {out}")


# ══════════════════════════════════════════════════════════════════════════
# CLASS 6 — FeatureExtractor
# ══════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    Wraps a FINE-TUNED EfficientNetB3 backbone and extracts 1536-dim
    embedding vectors for every image in a DataLoader.

    """

    def __init__(
        self,
        model:       "FineTunedEffNetB3",
        device:      Optional[torch.device] = None,
        batch_size:  int = 32,
        num_workers: int = 12,
    ) -> None:
        """
        Parameters
        ----------
        model       : Fine-tuned FineTunedEffNetB3 with the BEST checkpoint
                      already reloaded. Its classifier head is ignored;
                      extraction uses model.embed() (backbone → GAP →
                      flatten) only.
        device      : Auto-detected (CUDA if available, else CPU).
        batch_size  : Images per forward pass.
        num_workers : DataLoader worker processes.
        """
        self.device      = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.model       = self._prepare_model(model)
        # Callable that maps a batch of images → (batch, 1536) embeddings.
        # Kept named `backbone` so extract() below is unchanged.
        self.backbone    = self.model.embed

    def _prepare_model(self, model: "FineTunedEffNetB3") -> "FineTunedEffNetB3":
        """
        Move the fine-tuned model to device, switch to eval() mode and
        disable gradients on all parameters for deterministic, low-memory
        feature extraction.

        Returns
        -------
        FineTunedEffNetB3  Ready-to-extract model on self.device.
        """
        _header("STEP 4 — Preparing Fine-tuned EfficientNetB3 for Extraction")

        model = model.to(self.device).eval()

        # Disable gradients everywhere — extraction is a pure forward pass.
        for param in model.parameters():
            param.requires_grad = False

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Using fine-tuned EfficientNetB3 (blocks 6–8 specialised)")
        print(f"  Classifier head : ignored (embed() path only)")
        print(f"  Mode            : eval() — deterministic embeddings")
        print(f"  Parameters      : {n_params:,}  (gradients disabled)")
        print(f"  Output dim      : {EFFICIENTNET_B3_FEATURE_DIM}")
        print(f"  Device          : {self.device}")

        return model

    def extract(
        self,
        loader:     DataLoader,
        split_name: str = "split",
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Run a full forward pass through the backbone for every batch
        in loader and collect embedding vectors.

        Parameters
        ----------
        loader     : DataLoader  Must yield (image_tensor, label, image_id).
        split_name : str         For progress logging only.

        Returns
        -------
        features  : np.ndarray  (N, 1536)  float32
        labels    : np.ndarray  (N,)       int64
        image_ids : List[str]   length N   for metadata alignment
        """
        _header(f"STEP 5 — Extracting Features  [{split_name.upper()}]")
        print(f"  Batches : {len(loader)}")

        all_features:  List[np.ndarray] = []
        all_labels:    List[int]        = []
        all_image_ids: List[str]        = []

        with torch.no_grad():
            for i, (images, labels, image_ids) in enumerate(loader):
                images = images.to(self.device, non_blocking=True)

                # Forward pass — backbone is frozen so no backward graph
                # is constructed; this is purely a deterministic transform.
                embeddings = self.backbone(images)  # (B, 1536)

                all_features.append(
                    embeddings.cpu().numpy().astype(np.float32)
                )
                all_labels.extend(labels.numpy().tolist())
                all_image_ids.extend(list(image_ids))

                if (i + 1) % max(1, len(loader) // 10) == 0:
                    pct = (i + 1) / len(loader) * 100
                    print(f"  {i+1}/{len(loader)} batches  ({pct:.0f}%)")

        features = np.vstack(all_features)
        labels   = np.array(all_labels, dtype=np.int64)

        print(f"\n  Feature matrix : {features.shape}  dtype={features.dtype}")
        print(f"  Labels         : {labels.shape}")

        # Hard assertion — wrong output dim means backbone was built wrong
        assert features.shape[1] == EFFICIENTNET_B3_FEATURE_DIM, (
            f"Expected {EFFICIENTNET_B3_FEATURE_DIM} features, "
            f"got {features.shape[1]}"
        )

        return features, labels, all_image_ids


# ══════════════════════════════════════════════════════════════════════════
# CLASS 7 — FeatureStorageManager
# ══════════════════════════════════════════════════════════════════════════

class FeatureStorageManager:
    """
    Saves and loads feature matrices (.npy) and metadata CSVs.

    ─── FORMAT DESIGN DECISIONS ──────────────────────────────────────────
    Feature matrices → float32 NumPy .npy
        • np.load() memory-maps in O(1) — efficient for large arrays
        • float32 matches PyTorch default precision and halves memory
          vs float64 with no loss of information relevant to embeddings
        • Directly loadable as torch.tensor(np.load(...)) in Teammate C

    Metadata → CSV
        • Human-readable, universally importable
        • Preserves all column names from the DataFrame
        • image_id column enables explicit alignment verification
          (Teammate C should verify row i of features == row i of CSV
          by checking image_ids match, even though row order is guaranteed)

    ─── ALIGNMENT GUARANTEE ──────────────────────────────────────────────
    Row i of {split}_features.npy corresponds to row i of {split}_meta.csv.
    Both files are written from the same DataFrame at the same time.
    ExtractionPipeline also runs an explicit label-match assertion before
    calling save() as a final safety check.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        features: np.ndarray,
        meta_df:  pd.DataFrame,
        split:    str,
    ) -> None:
        """
        Persist feature matrix and metadata for one split.

        Parameters
        ----------
        features : np.ndarray   (N, 1536) float32
        meta_df  : pd.DataFrame N rows, aligned with features
        split    : "train" | "val" | "test"
        """
        feat_path = self.output_dir / f"{split}_features.npy"
        meta_path = self.output_dir / f"{split}_meta.csv"

        np.save(feat_path, features)
        meta_df.to_csv(meta_path, index=False)

        size_mb = features.nbytes / 1_048_576
        print(f"  [{split.upper():5s}]  "
              f"features → {feat_path.name}  "
              f"shape={features.shape}  ({size_mb:.1f} MB)")
        print(f"          "
              f"metadata → {meta_path.name}  "
              f"rows={len(meta_df)}")

    def load(
        self,
        split: str,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load and return (features, metadata) for a given split.

        Parameters
        ----------
        split : "train" | "val" | "test"

        Returns
        -------
        features : np.ndarray  (N, 1536)
        meta_df  : pd.DataFrame
        """
        feat_path = self.output_dir / f"{split}_features.npy"
        meta_path = self.output_dir / f"{split}_meta.csv"

        features = np.load(feat_path)
        meta_df  = pd.read_csv(meta_path)

        assert len(features) == len(meta_df), (
            f"Alignment error [{split}]: "
            f"features={len(features)}, meta={len(meta_df)}"
        )
        return features, meta_df

    def plot_feature_distributions(
        self,
        splits: Dict[str, np.ndarray],
    ) -> None:
        """
        Plot the per-feature mean activation distribution for each split.

        A well-behaved embedding shows:
            • Consistent distribution shape across train / val / test
              (divergence suggests a preprocessing inconsistency)
            • Mean near 0 (EfficientNet's BN layers centre activations)
            • No extreme outliers

        Parameters
        ----------
        splits : {"train": features_array, "val": ..., "test": ...}
        """
        _header("STEP 7 — Feature Distribution Diagnostics")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            "EfficientNetB3 Embedding Distribution per Split\n"
            "(histogram of per-feature mean activations across samples)",
            fontweight="bold",
        )

        for ax, (name, feat) in zip(axes, splits.items()):
            per_feat_mean = feat.mean(axis=0)
            ax.hist(per_feat_mean, bins=60, color="steelblue", edgecolor="none")
            ax.set_title(
                f"{name.upper()}  (N={feat.shape[0]:,})\n"
                f"mean={feat.mean():.3f}   std={feat.std():.3f}"
            )
            ax.set_xlabel("Per-feature mean activation")
            ax.set_ylabel("Feature count")

        plt.tight_layout()
        out = self.output_dir / "feature_distributions.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Feature distribution plot → {out}")

    def plot_class_distribution(
        self,
        train_meta: pd.DataFrame,
        val_meta:   pd.DataFrame,
        test_meta:  pd.DataFrame,
    ) -> None:
        """
        Bar chart of class counts per split — confirms augmentation
        targets were met and that no class is missing from any split.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            "HAM10000 Class Distribution After Augmentation",
            fontweight="bold",
        )
        class_names = [DX_LABEL_MAP[i] for i in range(NUM_CLASSES)]

        for ax, (name, meta) in zip(axes, [
            ("Train (augmented)", train_meta),
            ("Val (original)",    val_meta),
            ("Test (original)",   test_meta),
        ]):
            counts = [
                (meta["dx"] == c).sum() for c in range(NUM_CLASSES)
            ]
            ax.bar(class_names, counts, color="steelblue")
            ax.set_title(f"{name}\n(N={len(meta):,})")
            ax.set_xlabel("Class")
            ax.set_ylabel("Sample count")
            ax.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        out = self.output_dir / "class_distribution.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Class distribution plot   → {out}")


# ══════════════════════════════════════════════════════════════════════════
# CLASS 8 — ExtractionPipeline
# ══════════════════════════════════════════════════════════════════════════

class ExtractionPipeline:
    """
    Orchestrates the complete feature extraction workflow end-to-end.

    Execution sequence
    ──────────────────
        Step 1  Set global random seed
        Step 2  Compute proportional augmentation targets
        Step 3  Generate augmented training DataFrame (metadata rows only)
        Step 4  Build transforms (train + eval)
        Step 5  Fine-tune EfficientNetB3 (unfreeze blocks 6–8), monitor
                val loss, save + reload best checkpoint
        Step 6  Wrap the fine-tuned backbone in the FeatureExtractor
        Step 7  Build extraction DataLoaders (train_aug, val, test)
        Step 8  Extract features for all three splits
        Step 9  Verify label alignment (hard assertion)
        Step 10 Save .npy feature matrices + metadata CSVs
        Step 11 Plot diagnostics and print summary
    """

    def __init__(
        self,
        train_df:    pd.DataFrame,
        val_df:      pd.DataFrame,
        test_df:     pd.DataFrame,
        output_dir:  str   = "features",
        batch_size:  int   = 32,
        num_workers: int   = 12,
        random_seed: int   = 42,
    ) -> None:
        """
        Parameters
        ----------
        train_df    : Training DataFrame from images_cleaner.py.
        val_df      : Validation DataFrame.
        test_df     : Test DataFrame.
        output_dir  : Directory for feature files and diagnostic plots.
        batch_size  : Inference batch size.
        num_workers : DataLoader worker count.
        random_seed : Global seed for full reproducibility.
        """
        self.train_df    = train_df
        self.val_df      = val_df
        self.test_df     = test_df
        self.output_dir  = output_dir
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed

    def run(self) -> None:
        """Execute the full extraction pipeline."""
        _header("HAM10000 — EfficientNetB3 FEATURE EXTRACTION")
        print(f"  Random seed  : {self.random_seed}")
        print(f"  Output dir   : {self.output_dir}")
        print(f"  Batch size   : {self.batch_size}")
        _print_class_counts(self.train_df, "Train split (original)")

        # ── Step 1: Seed ───────────────────────────────────────────────
        _set_seed(self.random_seed)

        # ── Step 2: Device ─────────────────────────────────────────────
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"\n  Device: {device}")

        # ── Step 3: Augmentation targets + augmented DataFrame ─────────
        _header("STEP 1 — Proportional Class Balancing")
        aug_manager = AugmentationManager(random_seed=self.random_seed)
        targets     = aug_manager.compute_augmentation_targets(self.train_df)
        aug_train_df = aug_manager.generate_augmented_dataframe(
            self.train_df, targets
        )

        # ── Step 4: Transforms ─────────────────────────────────────────
        train_tf = aug_manager.build_train_transform()
        eval_tf  = aug_manager.build_eval_transform()

        # ── Step 5: Fine-tune EfficientNetB3 (blocks 6–8) ──────────────
        # Training uses the RAW train split with the stochastic train_tf
        # (fresh per-epoch augmentation), monitors validation loss, saves
        # the best checkpoint, and reloads it before any extraction.
        fine_tuner = FineTuner(
            train_df        = self.train_df,
            val_df          = self.val_df,
            train_transform = train_tf,
            eval_transform  = eval_tf,
            output_dir      = self.output_dir,
            device          = device,
            batch_size      = self.batch_size,
            num_workers     = self.num_workers,
        )
        finetuned_model = fine_tuner.fine_tune()   # best checkpoint reloaded

        # ── Step 6: Extractor (wraps the fine-tuned backbone) ──────────
        extractor = FeatureExtractor(
            model       = finetuned_model,
            device      = device,
            batch_size  = self.batch_size,
            num_workers = self.num_workers,
        )

        # ── Step 7: DataLoaders ────────────────────────────────────────
        # NOTE: extraction loaders use the EXPANDED augmented_train_df with
        # deterministic aug_params (shuffle=False) — preserving the
        # original export contract (originals + augmented rows).
        _header("STEP 4 — Building Extraction DataLoaders")
        ds_manager = DatasetManager(
            augmented_train_df = aug_train_df,
            val_df             = self.val_df,
            test_df            = self.test_df,
            train_transform    = train_tf,
            eval_transform     = eval_tf,
            batch_size         = self.batch_size,
            num_workers        = self.num_workers,
        )
        train_loader, val_loader, test_loader = ds_manager.build_loaders(device)

        # ── Step 8: Feature extraction (fine-tuned backbone) ───────────
        train_feat, train_labels, train_ids = extractor.extract(
            train_loader, "train"
        )
        val_feat,   val_labels,   val_ids   = extractor.extract(
            val_loader, "val"
        )
        test_feat,  test_labels,  test_ids  = extractor.extract(
            test_loader, "test"
        )

        # ── Step 9: Align metadata to extraction row order ─────────────
        # DataLoader used shuffle=False so row order is deterministic.
        # We reindex the DataFrames by image_id to match the extraction
        # sequence, then run a hard assertion before saving.
        train_meta = aug_train_df.set_index("image_id").loc[train_ids].reset_index()
        val_meta   = self.val_df.set_index("image_id").loc[val_ids].reset_index()
        test_meta  = self.test_df.set_index("image_id").loc[test_ids].reset_index()

        # Label alignment assertion — catches any DataFrame / DataLoader
        # ordering bug before the features are saved
        assert (train_meta["dx"].values == train_labels).all(), \
            "ALIGNMENT ERROR: train labels mismatch between features and metadata"
        assert (val_meta["dx"].values == val_labels).all(), \
            "ALIGNMENT ERROR: val labels mismatch"
        assert (test_meta["dx"].values == test_labels).all(), \
            "ALIGNMENT ERROR: test labels mismatch"
        print("\n  ✓ Label alignment verified for all three splits")

        # ── Step 10: Save ──────────────────────────────────────────────
        _header("STEP 6 — Saving Feature Matrices & Metadata")
        storage = FeatureStorageManager(self.output_dir)
        storage.save(train_feat, train_meta, "train")
        storage.save(val_feat,   val_meta,   "val")
        storage.save(test_feat,  test_meta,  "test")

        # ── Step 11: Diagnostics ───────────────────────────────────────
        storage.plot_feature_distributions({
            "train": train_feat,
            "val":   val_feat,
            "test":  test_feat,
        })
        storage.plot_class_distribution(train_meta, val_meta, test_meta)
        self._print_summary(train_feat, val_feat, test_feat)

        _header("EXTRACTION COMPLETE")
        print(f"\n  Output directory : {self.output_dir}/")
        print(f"  Feature files    : train/val/test_features.npy")
        print(f"  Metadata files   : train/val/test_meta.csv")
        print(f"\n  Teammate C: concatenate image embeddings (1536)")
        print(f"              with metadata vector (17) → 1553-dim input")

    @staticmethod
    def _print_summary(
        train_feat: np.ndarray,
        val_feat:   np.ndarray,
        test_feat:  np.ndarray,
    ) -> None:
        """Print a concise summary table of all extracted feature matrices."""
        _header("EXTRACTION SUMMARY")
        print(f"\n  {'Split':<6} {'N':>10} {'Dim':>6} "
              f"{'Mean':>8} {'Std':>8} {'MB':>8}")
        print(f"  {'-'*6} {'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
        for name, feat in [
            ("Train", train_feat),
            ("Val",   val_feat),
            ("Test",  test_feat),
        ]:
            mb = feat.nbytes / 1_048_576
            print(f"  {name:<6} {feat.shape[0]:>10,} {feat.shape[1]:>6} "
                  f"{feat.mean():>8.4f} {feat.std():>8.4f} {mb:>8.1f}")

        print(f"\n  Backbone        : EfficientNetB3 (IMAGENET1K_V1, "
              f"blocks 6–8 fine-tuned)")
        print(f"  Checkpoint      : {CHECKPOINT_NAME} (best val loss, reloaded)")
        print(f"  Input size      : {EFFICIENTNET_INPUT_SIZE}×{EFFICIENTNET_INPUT_SIZE}")
        print(f"  Embedding dim   : {EFFICIENTNET_B3_FEATURE_DIM}")
        print(f"  Metadata vector : age + sex + 15 OHE loc = 17 features")
        print(f"  Fusion input    : 1536 + 17 = 1553 features (for Teammate C)")


# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 68)
    print("  EffNetcnn.py — EfficientNetB3 Feature Extractor for HAM10000")
    print("  CSCI323 Modern Artificial Intelligence — Spring 2026, UOWD")
    print("=" * 68)

    # ============================================================
    # PATHS
    # ============================================================

    SPLIT_DIR = (
        r"C:\Users\goura\OneDrive\Desktop\Gourav Uni\CSCI323\Project\splitfiles"
    )

    OUTPUT_DIR = (
        r"C:\Users\goura\OneDrive\Desktop\Gourav Uni\CSCI323\Project\features"
    )

    # ============================================================
    # LOAD PRE-GENERATED SPLITS
    # Generated by images_cleaner.py
    # ============================================================

    print("\nLoading train/val/test split files...")

    train_df = pd.read_csv(
        os.path.join(SPLIT_DIR, "train_split.csv")
    )

    val_df = pd.read_csv(
        os.path.join(SPLIT_DIR, "val_split.csv")
    )

    test_df = pd.read_csv(
        os.path.join(SPLIT_DIR, "test_split.csv")
    )

    print(
        f"\nLoaded splits:"
        f"\n  Train : {len(train_df):,}"
        f"\n  Val   : {len(val_df):,}"
        f"\n  Test  : {len(test_df):,}"
    )

    # ============================================================
    # VERIFY FILEPATH COLUMN EXISTS
    # ============================================================

    required_columns = [
        "filepath",
        "image_id",
        "dx"
    ]

    for col in required_columns:
        if col not in train_df.columns:
            raise ValueError(
                f"Required column '{col}' missing from train_split.csv"
            )

    print("\n✓ Split files verified")

    # ============================================================
    # RUN FEATURE EXTRACTION
    # ============================================================

    pipeline = ExtractionPipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=OUTPUT_DIR,
        batch_size=32,
        num_workers=12,
        random_seed=42,
    )

    pipeline.run()

    # ============================================================
    # RELOAD AND VERIFY SAVED FILES
    # ============================================================

    print("\nVerification: reloading saved feature files...")

    storage = FeatureStorageManager(OUTPUT_DIR)

    for split in ["train", "val", "test"]:

        features, meta = storage.load(split)

        print(
            f"[{split.upper()}]"
            f" Features: {features.shape}"
            f" | Metadata: {meta.shape}"
            f" | Aligned: {len(features) == len(meta)}"
        )

    print("\nFeature extraction completed successfully.")
    print(f"Output directory: {OUTPUT_DIR}")
