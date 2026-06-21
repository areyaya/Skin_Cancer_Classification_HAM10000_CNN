"""
augmentation_runner.py
======================
Standalone runner for AugmentationManager (augG.py).

Run this AFTER images_cleaner.py and BEFORE the CNN pipeline.

Input  : train_split.csv  (produced by images_cleaner.py)
Output : augmented_train_split.csv  (expanded metadata rows, no pixel writes)

NOTE: torchvision is NOT required here. The transforms (build_train_transform /
      build_eval_transform) are used at training time inside your CNN Dataset
      class — not during metadata generation.

Usage
-----
    python augmentation_runner.py

Edit the paths in the CONFIG block below before running.
"""

# ── Imports ────────────────────────────────────────────────────────────────
import hashlib
import math
import os
import random
from typing import Dict, List, Tuple

import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — edit these paths before running
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_SPLIT_CSV  = r"C:\Users\flish\OneDrive\Desktop\SPRING2026\323\PROJ_FILRS\train_split.csv"
OUTPUT_CSV       = r"C:\Users\flish\OneDrive\Desktop\SPRING2026\323\PROJ_FILRS\augmented_train_split.csv"
RANDOM_SEED      = 42

# ── Shared label map (matches images_cleaner.py) ─────────────────────────
DX_LABEL_MAP: Dict[int, str] = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc",
}


def _header(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# AugmentationManager  (from augG.py)
# ══════════════════════════════════════════════════════════════════════════════

class AugmentationManager:
    """
    Generates an augmented training DataFrame with proportional class
    balancing using logarithmic scaling.

    Logarithmic scaling formula:
        scale(c)  = log(max_count + 1) / log(count(c) + 1)
        target(c) = clamp(ceil(count(c) × scale(c)),
                          MIN_AUGMENTED_SIZE, AUGMENTATION_CAP)

    Unique-combination constraint: no two augmented copies of the same
    original image share the same parameter tuple.
    """

    ROTATION_CHOICES:   List[int]   = list(range(0, 360, 10))
    ZOOM_CHOICES:       List[float] = [0.85, 0.88, 0.91, 0.94, 0.97, 1.00]
    BRIGHTNESS_CHOICES: List[float] = [0.80, 0.87, 0.93, 1.00, 1.07, 1.13, 1.20]
    CONTRAST_CHOICES:   List[float] = [0.80, 0.87, 0.93, 1.00, 1.07, 1.13, 1.20]

    MAX_RESAMPLE_ATTEMPTS: int = 200
    AUGMENTATION_CAP:      int = 8_000
    MIN_AUGMENTED_SIZE:    int = 300

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed   = random_seed
        self._seen_hashes: Dict[str, set] = {}

    # ── Proportional balancing ───────────────────────────────────────────

    def compute_augmentation_targets(self, df: pd.DataFrame) -> Dict[int, int]:
        """Compute per-class augmentation targets using logarithmic scaling."""
        counts    = df["dx"].value_counts().sort_index().to_dict()
        max_count = max(counts.values())

        print(f"\n  {'Class':<8} {'Current':>10} {'Target':>10}  {'Multiplier':>12}")
        print(f"  {'-'*8} {'-'*10} {'-'*10}  {'-'*12}")

        targets: Dict[int, int] = {}
        for cls, count in sorted(counts.items()):
            scale  = 1.0 if count >= max_count else (
                math.log(max_count + 1) / math.log(count + 1)
            )
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

        Each augmented row gets:
            image_id   = "{original_image_id}_aug_{N:03d}"
            aug_params = serialised parameter tuple for deterministic replay
        Original rows keep aug_params = None.
        """
        _header("STEP 2 — Generating Augmented Metadata Rows")

        df = df.copy()
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

                new_row               = dict(orig)
                new_row["image_id"]   = f"{img_id}_aug_{aug_counter[img_id]:03d}"
                new_row["aug_params"] = str(params)   # serialise tuple → string for CSV
                new_rows.append(new_row)

        aug_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        print(f"\n  Original rows   : {len(df):,}")
        print(f"  Synthetic rows  : {len(new_rows):,}")
        print(f"  Augmented total : {len(aug_df):,}")
        return aug_df

    # ── Unique parameter sampling ────────────────────────────────────────

    def _sample_unique_params(self, image_id: str) -> Tuple:
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
        return params


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    random.seed(RANDOM_SEED)

    _header("AUGMENTATION RUNNER — START")
    print(f"  Input  : {TRAIN_SPLIT_CSV}")
    print(f"  Output : {OUTPUT_CSV}")

    # ── Load train split ─────────────────────────────────────────────────
    _header("STEP 1 — Loading Train Split")
    train_df = pd.read_csv(TRAIN_SPLIT_CSV)
    print(f"  Loaded {len(train_df):,} rows")
    print(f"  Columns: {list(train_df.columns)}")

    # ── Run augmentation ─────────────────────────────────────────────────
    manager = AugmentationManager(random_seed=RANDOM_SEED)

    _header("STEP 1b — Computing Augmentation Targets")
    targets = manager.compute_augmentation_targets(train_df)

    aug_train_df = manager.generate_augmented_dataframe(train_df, targets)

    # ── Save augmented metadata ──────────────────────────────────────────
    _header("STEP 3 — Saving Augmented Train Split")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    aug_train_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved {len(aug_train_df):,} rows → {OUTPUT_CSV}")

    # ── Final class distribution ─────────────────────────────────────────
    _header("FINAL — Augmented Class Distribution")
    print(f"\n  {'Class':<8} {'Rows':>8}")
    print(f"  {'-'*8} {'-'*8}")
    for code, name in DX_LABEL_MAP.items():
        n = (aug_train_df["dx"] == code).sum()
        print(f"  {name:<8} {n:>8,}")
    print(f"\n  Total: {len(aug_train_df):,}")
    print("\n  Done. Pass augmented_train_split.csv into your CNN pipeline.")
    print("  Val/test splits are unchanged — use the originals as-is.")
