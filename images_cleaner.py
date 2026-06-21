"""
images_cleaner.py
=================
CSCI323 Modern Artificial Intelligence - Spring 2026
University of Wollongong Dubai (UOWD)

Dataset : HAM10000
          ~9,958 dermoscopy images across two folders (part1 / part2)
          + HAM10000_metadata_cleaned.csv

Purpose
-------
This file is a PURE PREPROCESSING AND DATASET INTEGRITY MODULE.

It does exactly five things and nothing else:
    1. Load and validate the cleaned metadata CSV
    2. Scan image folders and verify every metadata row has a real
       image file on disk (and vice versa)
    3. Enforce a strict 1:1 mapping between image_id and image file —
       no duplicate rows, no missing paths, no orphan files
    4. One-hot encode the 'localization' column (fitted on train only
       to prevent data leakage into val/test) into 15 binary columns
       named loc_<site> (e.g. loc_abdomen, loc_back, ...)
    5. Split into train / val / test on lesion_id with stratification
       on dx

THIS FILE MUST NEVER:
    - Modify pixel data
    - Create new image files
    - Generate augmented or synthetic images
    - Increase dataset size artificially
    - Perform any class balancing

Augmentation is handled separately in EffNetcnn.py using on-the-fly
PyTorch Dataset transforms. Keeping augmentation out of this file
ensures clean separation of concerns and prevents any possibility of
augmented data leaking into val or test sets.

Metadata columns (as received)
--------------------------------
    lesion_id    — unique ID per skin lesion (multiple images can share)
    image_id     — unique ID per image → maps directly to filename on disk
    dx           — label-encoded diagnosis target (int 0-6):
                       0=akiec  1=bcc  2=bkl  3=df  4=mel  5=nv  6=vasc
    age          — patient age (int, already imputed in data_cleaner.py)
    sex          — binary encoded: 0=male, 1=female
    localization — body site string (15 categories, raw — OHE encoded here)

Split level — WHY lesion_id and NOT image_id
---------------------------------------------
HAM10000 contains 1,951 lesions with more than one image (different
angles / lighting of the same patch of skin). If we split at the
image level, images of the same lesion can appear in BOTH train and
test. The model would be tested on lesions it has already seen,
making the evaluation a measure of memorisation, not generalisation.

Splitting at the lesion level ensures that every image of a given
lesion lands in exactly one split. The model is evaluated only on
lesions it has never seen in any form.

OOP structure
-------------
    ImageRegistry   — scans image folders, builds image_id → path dict
    DatasetCleaner  — validates integrity, resolves duplicates,
                      produces a clean dataframe
    DatasetSplitter — lesion-aware stratified train / val / test split
                      + OHE encoding of localization (fit on train only)
    PipelineRunner  — orchestrates all three classes end-to-end

Output
------
    train_df, val_df, test_df  — three clean DataFrames, each containing
    only real images, with a strict 1:1 image_id → file mapping, ready
    to be passed into a PyTorch Dataset class in EffNetcnn.py.

    localization is expanded into 15 binary columns:
        loc_abdomen, loc_acral, loc_back, loc_chest, loc_ear,
        loc_face, loc_foot, loc_genital, loc_hand,
        loc_lower extremity, loc_neck, loc_scalp, loc_trunk,
        loc_unknown, loc_upper extremity

    These 15 columns + age + sex form the metadata feature vector
    that is concatenated with the EfficientNetB3 image embeddings.
    dx remains a single label-encoded integer (0–6) — it is the
    target label, NOT an input feature.

Usage
-----
    from images_cleaner import PipelineRunner

    runner = PipelineRunner(
        metadata_path = "HAM10000_metadata_cleaned.csv",
        image_dirs    = ["HAM10000_images_part1", "HAM10000_images_part2"],
    )
    train_df, val_df, test_df = runner.run()
"""


# ---------- IMPORTS ----------
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ---------- PLOT STYLE ----------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120})

# ---------- DX LABEL MAP ----------
# 'dx' is already label-encoded in the cleaned CSV (int 0-6).
# This map is used only for logging and plot labels — never for encoding.
DX_LABEL_MAP = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc"
}


# ─────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────
def _header(title: str) -> None:
    """Print a section divider to stdout for readability."""
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def _print_class_counts(df: pd.DataFrame, label: str) -> None:
    """
    Print the count of each dx class in df.

    Parameters
    ----------
    df    : pd.DataFrame  Any dataframe that has a 'dx' column.
    label : str           Description shown before the counts.
    """
    print(f"\n  {label}:")
    for code, name in DX_LABEL_MAP.items():
        n = (df["dx"] == code).sum()
        print(f"    {code} ({name:8s}): {n:,}")


# ═════════════════════════════════════════════════════════════════════
# CLASS 1 — ImageRegistry
# ═════════════════════════════════════════════════════════════════════
class ImageRegistry:
    """
    Scans one or more image folders and builds a lookup dictionary:

        { image_id : absolute_file_path }

    HAM10000 images live in two flat folders (part1, part2). Every
    file is named exactly after its image_id (e.g. 'ISIC_0027419.jpg'
    -> image_id 'ISIC_0027419'). There are no subdirectories.

    Keeping I/O in its own class means no downstream class ever needs
    to know where images live on disk. If the folder layout changes,
    only this class needs updating.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, image_dirs: list):
        """
        Parameters
        ----------
        image_dirs : list[str]
            Folder paths to scan. Supports one folder or many.
        """
        self.image_dirs = image_dirs
        self.registry   = {}   # { image_id (str) : absolute_path (str) }

    def build(self) -> dict:
        """
        Walk every folder in self.image_dirs and populate self.registry.

        Returns
        -------
        dict
            { image_id : absolute_file_path }
        """
        _header("STEP 1 — Building Image Registry")

        for folder in self.image_dirs:
            if not os.path.isdir(folder):
                # Warn but continue — user may have only one image folder.
                print(f"  WARNING — folder not found, skipping: {folder}")
                continue

            count = 0
            for fname in os.listdir(folder):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.SUPPORTED_EXTENSIONS:
                    continue

                # Strip extension to get image_id.
                # 'ISIC_0027419.jpg' → 'ISIC_0027419'
                image_id  = os.path.splitext(fname)[0]
                full_path = os.path.abspath(os.path.join(folder, fname))
                self.registry[image_id] = full_path
                count += 1

            print(f"  Scanned: {folder}  →  {count:,} images indexed")

        print(f"\n  Total images in registry: {len(self.registry):,}")

        # Hard assertion — if registry is empty, nothing downstream
        # will work. Fail loudly here rather than silently later.
        assert len(self.registry) > 0, \
            "Registry is empty. Check that image_dirs paths are correct."

        return self.registry


# ═════════════════════════════════════════════════════════════════════
# CLASS 2 — DatasetCleaner
# ═════════════════════════════════════════════════════════════════════
class DatasetCleaner:
    """
    Validates dataset integrity and produces a single clean DataFrame.

    Responsibilities
    ─────────────────
    load_metadata()
        Load the CSV and print an initial profile.

    verify_images_exist()
        Cross-reference the metadata against the image registry.
        Drop any metadata row where the image file does not exist.
        Log any image files that have no metadata row (orphans).

    clean_dataset()
        Enforce strict 1:1 mapping between image_id and image file:
            - Resolve any duplicate image_id rows (keep first)
            - Assert no missing filepaths remain
            - Assert no duplicate image_ids remain
            - Assert every filepath still exists on disk

        NOTE: 'localization' is NOT encoded here. Encoding is deferred
        until after the train/val/test split in DatasetSplitter so
        that the OneHotEncoder is fitted on train data only, preventing
        any data leakage from val/test distributions.

    The clean DataFrame has one row per image_id with columns:
        lesion_id | image_id | filepath | dx | age | sex | localization
    where localization is still the raw string at this stage.
    """

    def __init__(self, metadata_path: str, registry: dict):
        """
        Parameters
        ----------
        metadata_path : str   Path to HAM10000_metadata_cleaned.csv.
        registry      : dict  Output of ImageRegistry.build().
        """
        self.metadata_path = metadata_path
        self.registry      = registry

    # ── PUBLIC METHOD 1 ───────────────────────────────────────────
    def load_metadata(self) -> pd.DataFrame:
        """
        Load the metadata CSV and print a full profile.

        Returns
        -------
        pd.DataFrame
            Raw metadata as loaded from disk — no modifications.
        """
        _header("STEP 2a — Loading Metadata")

        df = pd.read_csv(self.metadata_path)

        print(f"  File   : {self.metadata_path}")
        print(f"  Shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n  Dtypes:\n{df.dtypes.to_string()}")
        print(f"\n  Missing values:\n{df.isnull().sum().to_string()}")
        print(f"\n  Duplicate image_ids: {df['image_id'].duplicated().sum()}")
        print(f"  Duplicate lesion_ids (expected — multi-image lesions): "
              f"{df['lesion_id'].duplicated().sum()}")

        _print_class_counts(df, "Class distribution (raw)")
        return df

    # ── PUBLIC METHOD 2 ───────────────────────────────────────────
    def verify_images_exist(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-reference metadata against the image registry.

        Three checks:
            1. Metadata rows with no matching image file → dropped.
               We cannot use a row if there is no image to load.

            2. Image files with no matching metadata row → logged.
               These are excluded from the dataframe but not deleted
               from disk — this file never modifies the filesystem.

            3. A 'filepath' column is added to df for every matched row.

        Parameters
        ----------
        df : pd.DataFrame  Raw metadata from load_metadata().

        Returns
        -------
        pd.DataFrame
            df with 'filepath' column added; unmatched rows removed.
        """
        _header("STEP 2b - Verifying Image-Metadata Alignment")

        before = len(df)

        # ── Map every image_id to its filepath via the registry ────
        # Rows where image_id is not in the registry will get NaN.
        df = df.copy()
        df["filepath"] = df["image_id"].map(self.registry)

        # ── Drop rows with no matching image file ──────────────────
        # NaN filepath means the image does not exist on disk.
        missing_img = df["filepath"].isna().sum()
        if missing_img > 0:
            print(f"  Metadata rows with no image file: {missing_img:,} → dropped")
            df = df.dropna(subset=["filepath"]).reset_index(drop=True)
            
        else:
            print(f"  All metadata rows have a matching image file")

        after = len(df)
        print(f"  Metadata rows: {before:,} → {after:,}  "
              f"(dropped {before - after:,})")

        # ── Log orphan images (in registry but not in metadata) ────
        # These are NOT added to df — any image without a metadata row
        # is excluded per project spec.
        meta_ids    = set(df["image_id"])
        reg_ids     = set(self.registry.keys())
        orphan_imgs = reg_ids - meta_ids
        if orphan_imgs:
            count = 0
            print(f"\n  Images with no metadata row: {len(orphan_imgs):,}")
            for image_id in sorted(orphan_imgs):
                count += 1
                print(f" Image number {count} with ID: {image_id}")
                print(f" Path {count} for ID {image_id}: {self.registry[image_id]}")
        else:
            print(f"  All registry images have a metadata row")

        print(f"\n  Coverage: {len(df):,} image-metadata pairs confirmed")
        return df

    # ── PUBLIC METHOD 3 ───────────────────────────────────────────
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce strict 1:1 mapping between image_id and image file,
        then run integrity assertions to confirm a clean state.

        Steps
        ──────
        1. Resolve duplicate image_id rows (keep first occurrence).
           WHY keep first? The first row is the original record in the
           CSV as it was cleaned. Later duplicates are more likely to
           be artefacts of a merge or re-index operation.

        2. Assert no duplicate image_ids remain.
        3. Assert no missing filepaths remain.
        4. Assert every filepath stored in the dataframe actually
           exists on disk right now (guards against files moved/deleted
           between registry scan and this point in the pipeline).

        NOTE: 'localization' is NOT encoded here. Encoding is deferred
        until after the train/val/test split in DatasetSplitter so
        that the OneHotEncoder is fitted on train data only.

        Parameters
        ----------
        df : pd.DataFrame  Output of verify_images_exist().

        Returns
        -------
        pd.DataFrame
            Fully clean DataFrame with strict 1:1 image_id -> filepath.
        """
        _header("STEP 2c — Enforcing Dataset Integrity")

        df = df.copy()
        before = len(df)

        # ── Resolve duplicate image_ids ────────────────────────────
        # image_id must uniquely identify one image file. Any row
        # sharing an image_id with an earlier row is a duplicate.
        n_dupes = df["image_id"].duplicated().sum()
        if n_dupes > 0:
            print(f"  Duplicate image_id rows found: {n_dupes:,} -> keeping first")
            df = df.drop_duplicates(subset=["image_id"], keep="first")
            df = df.reset_index(drop=True)
        else:
            print(f"  No duplicate image_id rows found  -> no action needed/taken")

        after_dedup = len(df)
        print(f"  Rows after deduplication: {after_dedup:,}  "
              f"(removed {before - after_dedup:,})")

        # ── Integrity assertions ───────────────────────────────────
        # These will raise AssertionError immediately if violated,
        # preventing a corrupted dataset from reaching the split step.

        # 1. No duplicate image_ids
        assert df["image_id"].duplicated().sum() == 0, \
            "INTEGRITY FAIL: duplicate image_ids remain after deduplication"

        # 2. No missing filepaths
        assert df["filepath"].isna().sum() == 0, \
            "INTEGRITY FAIL: NaN filepaths remain — all rows must have an image"

        # 3. Every filepath on disk right now
        # We check a sample if the dataset is large to avoid slow I/O,
        # but for HAM10000 (~10k rows) a full check is fast enough.
        bad_paths = [p for p in df["filepath"] if not os.path.isfile(p)]
        assert len(bad_paths) == 0, \
            (f"INTEGRITY FAIL: {len(bad_paths)} filepaths point to files that "
             f"no longer exist on disk. First: {bad_paths[0]}")

        print(f"\n  All integrity checks passed")
        print(f"  Final clean dataset: {len(df):,} rows")
        print(f"  Unique image_ids  : {df['image_id'].nunique():,}")
        print(f"  Unique lesion_ids : {df['lesion_id'].nunique():,}")

        _print_class_counts(df, "Class distribution (after cleaning)")
        self._plot_class_distribution(df)
        return df

    def _plot_class_distribution(self, df: pd.DataFrame) -> None:
        """
        Bar chart of class distribution in the clean dataset.
        This shows the true baseline imbalance before any split.
        """
        vals    = [df[df["dx"] == i].shape[0] for i in sorted(DX_LABEL_MAP)]
        labels  = [DX_LABEL_MAP[i] for i in sorted(DX_LABEL_MAP)]
        palette = sns.color_palette("muted", len(labels))

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, vals, color=palette, edgecolor="white")
        ax.set_title("Class Distribution — Clean Dataset\n"
                     "(real images only, no augmentation)",
                     fontweight="bold")
        ax.set_xlabel("Diagnosis Class")
        ax.set_ylabel("Image Count")
        ax.axhline(sum(vals) / len(vals), color="red", linestyle="--",
                   linewidth=1.2, label=f"Mean: {sum(vals)/len(vals):.0f}")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 30, str(v), ha="center", fontsize=9)
        ax.legend()
        plt.tight_layout()
        plt.savefig("plot_clean_class_dist.png", bbox_inches="tight")
        plt.show()
        print("\n  [Plot] Clean class distribution → plot_clean_class_dist.png")


# ═════════════════════════════════════════════════════════════════════
# CLASS 3 — DatasetSplitter
# ═════════════════════════════════════════════════════════════════════
class DatasetSplitter:
    """
    Performs a stratified train / val / test split, then one-hot
    encodes the 'localization' column using an OneHotEncoder fitted
    on train only.

    WHY split on lesion_id and NOT image_id
    ─────────────────────────────────────────
    HAM10000 contains 1,951 lesions with more than one image. If we
    split at the image level, different images of the same lesion
    (same patch of skin, different angles / lighting) can appear on
    both sides of the train/test boundary.

    The model would then be tested on lesions it has already seen
    during training. The test score would measure memorisation of
    lesion-specific texture, not generalisation to unseen patients.
    This is data leakage.

    Splitting at the lesion level ensures every image of a given
    lesion lands in exactly one split. No lesion crosses a split
    boundary.

    Stratification
    ──────────────
    Stratification is applied at the LESION level using each lesion's
    dx label. Every HAM10000 lesion has exactly one dx class. This
    ensures that rare classes (df: 115 images, vasc: 142) are
    proportionally represented in all three splits.

    localization OHE encoding
    ──────────────────────────
    OneHotEncoder is fitted ONLY on the train set's localization values,
    then used to transform val and test. This prevents data leakage:
    the encoding is determined solely from training data.

    WHY OneHotEncoder and not LabelEncoder?
        localization is a nominal (unordered) categorical variable —
        there is no meaningful ordering between body sites like 'back'
        and 'scalp'. OneHotEncoder represents each site as an
        independent binary column, so no integer distance between sites
        is implied. This is the correct encoding for a feature being
        fed into a neural network dense layer.

        Each of the 15 body sites becomes its own binary column:
            loc_abdomen, loc_acral, loc_back, ... loc_upper extremity

        For any given row exactly one of these columns is 1, the rest
        are 0. This is what the metadata dense branch of EfficientNetB3
        will receive as input alongside age and sex.

    WHY keep all 15 columns (no drop='first')?
        drop='first' removes one column to avoid perfect multicollinearity,
        which matters for linear models (OLS). Neural networks do not
        suffer from multicollinearity — the network learns its own
        internal weighting. Keeping all 15 columns also preserves full
        interpretability: every site has a clearly named column.

    The fitted encoder is stored in self.loc_encoder and exposed
    publicly so EffNetcnn.py can:
        - Know which column index corresponds to which site name
          via loc_encoder.categories_[0]
        - Encode new inference-time localization strings consistently
    """

    def __init__(self,
                 train_size:   float = 0.70,
                 val_size:     float = 0.15,
                 test_size:    float = 0.15,
                 random_state: int   = 42):
        """
        Parameters
        ----------
        train_size   : float  Fraction of lesions for training.
        val_size     : float  Fraction of lesions for validation.
        test_size    : float  Fraction of lesions for testing.
        random_state : int    Seed — ensures the same split every run.
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-9, \
            "train_size + val_size + test_size must equal 1.0"

        self.train_size   = train_size
        self.val_size     = val_size
        self.test_size    = test_size
        self.random_state = random_state

        # OneHotEncoder — fitted on train localization values only.
        # sparse_output=False  → returns a dense numpy array, not a
        #                        sparse matrix, so it can be inserted
        #                        directly into a pandas DataFrame.
        # handle_unknown='ignore' → if val/test contains a site not
        #                           seen in train, all OHE columns for
        #                           that row are set to 0 rather than
        #                           raising an error.
        self.loc_encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            dtype=np.int8           # binary 0/1 stored as int8 to save memory
        )

    def split(self, df: pd.DataFrame) -> tuple:
        """
        Split df into train / val / test, then OHE-encode localization.

        Strategy
        ─────────
        1. Collapse to one row per lesion_id (each lesion has one dx).
        2. First split: carve out the test set (15%).
        3. Second split: divide the remaining 85% into val (approx. 17.6%)
           and train — producing the correct final 70/15/15 fractions.
        4. Expand back to image level using the split lesion ID sets.
        5. Fit OneHotEncoder on train['localization'], transform all three.

        Parameters
        ----------
        df : pd.DataFrame  Clean DataFrame from DatasetCleaner.

        Returns
        -------
        tuple : (train_df, val_df, test_df)
            Each is a subset of the clean df rows.
            'localization' column replaced by 15 binary loc_<site> columns.
        """
        _header("STEP 3 — Stratified Train / Val / Test Split")

        # ── 3a. Build lesion-level table ───────────────────────────
        # One row per lesion with its dx label.
        # Every lesion in HAM10000 has a single consistent dx value.
        lesion_df = (df.groupby("lesion_id")["dx"]
                       .first()
                       .reset_index())

        print(f"  Total unique lesions : {len(lesion_df):,}")
        print(f"  Total images         : {len(df):,}")
        print(f"  Target split         : "
              f"{self.train_size:.0%} / {self.val_size:.0%} / {self.test_size:.0%}")

        # ── 3b. First split — carve out test (15%) ─────────────────
        train_val_lesions, test_lesions = train_test_split(
            lesion_df,
            test_size    = self.test_size,
            stratify     = lesion_df["dx"],
            random_state = self.random_state
        )

        # ── 3c. Second split — carve val from train+val ─────────────
        # val fraction relative to the train+val pool only:
        # 0.15 / (0.70 + 0.15) = 0.1765...
        val_frac = self.val_size / (self.train_size + self.val_size)

        train_lesions, val_lesions = train_test_split(
            train_val_lesions,
            test_size    = val_frac,
            stratify     = train_val_lesions["dx"],
            random_state = self.random_state
        )

        # ── 3d. Expand back to image level ─────────────────────────
        train_ids = set(train_lesions["lesion_id"])
        val_ids   = set(val_lesions["lesion_id"])
        test_ids  = set(test_lesions["lesion_id"])

        train_df = df[df["lesion_id"].isin(train_ids)].copy().reset_index(drop=True)
        val_df   = df[df["lesion_id"].isin(val_ids)].copy().reset_index(drop=True)
        test_df  = df[df["lesion_id"].isin(test_ids)].copy().reset_index(drop=True)

        # ── 3e. Verify no lesion crosses a split boundary ──────────
        assert train_ids.isdisjoint(val_ids),  \
            "SPLIT FAIL: lesion_ids overlap between train and val"
        assert train_ids.isdisjoint(test_ids), \
            "SPLIT FAIL: lesion_ids overlap between train and test"
        assert val_ids.isdisjoint(test_ids),   \
            "SPLIT FAIL: lesion_ids overlap between val and test"

        # ── 3f. Verify no image_id appears in more than one split ──
        all_img_ids = (set(train_df["image_id"]) |
                       set(val_df["image_id"])   |
                       set(test_df["image_id"]))
        assert len(all_img_ids) == len(train_df) + len(val_df) + len(test_df), \
            "SPLIT FAIL: image_ids are not unique across splits"

        # ── 3g. OHE-encode localization ────────────────────────────
        train_df, val_df, test_df = self._encode_localization(
            train_df, val_df, test_df
        )

        # ── 3h. Log results ────────────────────────────────────────
        total = len(df)
        for name, sdf in [("Train", train_df),
                          ("Val",   val_df),
                          ("Test",  test_df)]:
            pct = len(sdf) / total * 100
            print(f"\n  {name}: {len(sdf):,} images  ({pct:.1f}%)  "
                  f"| {sdf['lesion_id'].nunique():,} lesions")
            for code in sorted(DX_LABEL_MAP):
                n = (sdf["dx"] == code).sum()
                print(f"    {code} ({DX_LABEL_MAP[code]:8s}): {n:,}")

        self._plot_split_distribution(train_df, val_df, test_df)
        return train_df, val_df, test_df

    def _encode_localization(self,
                             train_df: pd.DataFrame,
                             val_df:   pd.DataFrame,
                             test_df:  pd.DataFrame) -> tuple:
        """
        One-hot encode the 'localization' column.

        Steps
        ──────
        1. Fit OneHotEncoder on train['localization'] only.
           Fitting on the full dataset before splitting would
           constitute data leakage because the encoder would learn
           the distribution of val/test site values.

        2. Transform all three splits using the fitted encoder.
           handle_unknown='ignore' means any site in val/test that was
           not seen in train produces an all-zero row for that sample,
           rather than raising an error.

        3. Drop the original raw 'localization' string column.

        4. Expand the OHE output into 15 named binary columns:
               loc_abdomen, loc_acral, loc_back, loc_chest, loc_ear,
               loc_face, loc_foot, loc_genital, loc_hand,
               loc_lower extremity, loc_neck, loc_scalp, loc_trunk,
               loc_unknown, loc_upper extremity

           Column names are derived from the encoder's fitted
           categories (loc_encoder.categories_[0]), so they always
           match the actual values seen during fitting — no hardcoding.

        5. Concatenate the new binary columns back onto the DataFrame
           at the end, after the existing columns.

        The fitted encoder is stored in self.loc_encoder.
        Access the site-name → column mapping in EffNetcnn.py via:
            runner.loc_encoder.categories_[0]  → array of site names
            (index i in this array corresponds to column loc_<name>)

        Parameters
        ----------
        train_df, val_df, test_df : pd.DataFrame
            Each must contain a raw string 'localization' column.

        Returns
        -------
        tuple : (train_df, val_df, test_df)
            'localization' string column removed.
            15 binary loc_<site> columns appended.
        """
        _header("STEP 3b — One-Hot Encoding Localization (fit on train only)")

        # ── Fit on train localization values only ──────────────────
        # reshape(-1, 1) is required because OneHotEncoder expects a
        # 2D array input: (n_samples, n_features). Our single column
        # is 1D, so we reshape to (n_samples, 1).
        self.loc_encoder.fit(train_df[["localization"]])

        # ── Derive column names from fitted categories ─────────────
        # loc_encoder.categories_[0] is a sorted array of all unique
        # localization strings seen during fit, e.g.:
        #   ['abdomen', 'acral', 'back', ..., 'upper extremity']
        # We prefix each with 'loc_' for clarity in the DataFrame.
        site_names   = self.loc_encoder.categories_[0]
        ohe_col_names = [f"loc_{site}" for site in site_names]

        print(f"  OneHotEncoder fitted on train set.")
        print(f"  {len(ohe_col_names)} localization columns created:")
        for i, col in enumerate(ohe_col_names):
            print(f"    [{i:2d}] {col}")

        # ── Transform each split and expand into named columns ─────
        def _transform(df: pd.DataFrame) -> pd.DataFrame:
            """
            Apply the fitted OHE to df['localization'] and return
            a new DataFrame with the string column replaced by
            15 binary loc_<site> columns.

            Parameters
            ----------
            df : pd.DataFrame  Must contain raw string 'localization'.

            Returns
            -------
            pd.DataFrame
                Original columns (minus 'localization') + 15 loc_ columns.
            """
            # Transform — produces a (n_rows, 15) numpy array of int8
            ohe_array = self.loc_encoder.transform(df[["localization"]])

            # Build a DataFrame from the OHE output with proper column names
            ohe_df = pd.DataFrame(
                ohe_array,
                columns=ohe_col_names,
                index=df.index        # align index so concat works correctly
            )

            # Drop the raw string column and append the 15 binary columns
            df = df.drop(columns=["localization"])
            df = pd.concat([df, ohe_df], axis=1)
            return df

        train_df = _transform(train_df)
        val_df   = _transform(val_df)
        test_df  = _transform(test_df)

        # ── Verify shape: each df should now have 15 extra columns ─
        assert all(col in train_df.columns for col in ohe_col_names), \
            "OHE FAIL: not all loc_ columns present in train_df"

        print(f"\n  Encoding complete.")
        print(f"  'localization' string column removed.")
        print(f"  {len(ohe_col_names)} binary loc_ columns appended.")
        print(f"\n  To map column index back to site name in EffNetcnn.py:")
        print(f"    runner.loc_encoder.categories_[0]  "
              f"→ array of {len(site_names)} site names")

        return train_df, val_df, test_df


    def _plot_split_distribution(self,
                                 train_df: pd.DataFrame,
                                 val_df:   pd.DataFrame,
                                 test_df:  pd.DataFrame) -> None:
        """
        Side-by-side bar charts of the class distribution per split.
        If stratification worked, the class ratios should be
        approximately proportional across all three charts.
        """
        splits  = {"Train": train_df, "Val": val_df, "Test": test_df}
        labels  = [DX_LABEL_MAP[i] for i in sorted(DX_LABEL_MAP)]
        palette = sns.color_palette("muted", len(labels))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        fig.suptitle("Class Distribution per Split\n"
                     "(lesion-level stratification, real images only)",
                     fontweight="bold")

        for ax, (name, sdf) in zip(axes, splits.items()):
            vals = [sdf[sdf["dx"] == i].shape[0]
                    for i in sorted(DX_LABEL_MAP)]
            ax.bar(labels, vals, color=palette, edgecolor="white")
            ax.set_title(f"{name}  (n={len(sdf):,})")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=30)
            for i, v in enumerate(vals):
                ax.text(i, v + 1, str(v), ha="center", fontsize=7)

        plt.tight_layout()
        plt.savefig("plot_split_distribution.png", bbox_inches="tight")
        plt.show()
        print("\n  [Plot] Split distribution → plot_split_distribution.png")


# ═════════════════════════════════════════════════════════════════════
# CLASS 4 — PipelineRunner
# ═════════════════════════════════════════════════════════════════════
class PipelineRunner:
    """
    Orchestrates all three classes end-to-end and returns
    train_df, val_df, test_df ready for EffNetcnn.py.

    This is the only class a caller needs to interact with.

    Final DataFrame columns
    ────────────────────────
        lesion_id      — original lesion identifier (string)
        image_id       — unique image identifier (string)
        filepath       — absolute path to the image on disk (string)
        dx             — label-encoded diagnosis TARGET (int 0-6)
                         NOT an input feature — passed to loss function
        age            — patient age (int) — metadata input feature
        sex            — binary: 0=male, 1=female (int) — metadata feature
        loc_abdomen    — OHE binary (int8) — metadata input feature
        loc_acral      — OHE binary (int8)
        loc_back       — OHE binary (int8)
        loc_chest      — OHE binary (int8)
        loc_ear        — OHE binary (int8)
        loc_face       — OHE binary (int8)
        loc_foot       — OHE binary (int8)
        loc_genital    — OHE binary (int8)
        loc_hand       — OHE binary (int8)
        loc_lower extremity — OHE binary (int8)
        loc_neck       — OHE binary (int8)
        loc_scalp      — OHE binary (int8)
        loc_trunk      — OHE binary (int8)
        loc_unknown    — OHE binary (int8)
        loc_upper extremity — OHE binary (int8)

    Metadata feature vector for EfficientNetB3 dense branch:
        [ age, sex, loc_abdomen, loc_acral, ..., loc_upper extremity ]
        = 2 numeric + 15 binary = 17 features total

    Accessing the localization category list in EffNetcnn.py
    ─────────────────────────────────────────────────────────
        runner = PipelineRunner(...)
        train_df, val_df, test_df = runner.run()

        # Full ordered list of site names (matches column order):
        sites = runner.loc_encoder.categories_[0]
        # sites[0] == 'abdomen' → corresponds to column 'loc_abdomen'
    """

    def __init__(self,
                 metadata_path: str,
                 image_dirs:    list,
                 train_size:    float = 0.70,
                 val_size:      float = 0.15,
                 test_size:     float = 0.15,
                 random_state:  int   = 42):
        """
        Parameters
        ----------
        metadata_path : str         Path to HAM10000_metadata_cleaned.csv.
        image_dirs    : list[str]   Image folder paths to scan.
        train_size    : float       Fraction for training.
        val_size      : float       Fraction for validation.
        test_size     : float       Fraction for testing.
        random_state  : int         Global random seed for reproducibility.
        """
        self.metadata_path = metadata_path
        self.image_dirs    = image_dirs
        self.train_size    = train_size
        self.val_size      = val_size
        self.test_size     = test_size
        self.random_state  = random_state

        # Exposed after run() — use categories_[0] to map column
        # indices back to site name strings in EffNetcnn.py
        self.loc_encoder: OneHotEncoder | None = None

    def run(self) -> tuple:
        """
        Execute the full pipeline and return the three split DataFrames.

        Returns
        -------
        tuple : (train_df, val_df, test_df)
            Each DataFrame contains only real images.
            No augmented or synthetic data.
            Strict 1:1 image_id → filepath mapping.
            'localization' replaced by 15 binary loc_<site> columns.
            Ready to be passed into a PyTorch Dataset class.
        """
        _header("HAM10000 IMAGES CLEANER — START")
        print("  Purpose : dataset integrity + splitting only")
        print("  Augmentation : NONE (handled in EffNetcnn.py)")

        # ── Step 1: Build image registry ──────────────────────────
        registry = ImageRegistry(self.image_dirs).build()

        # ── Step 2: Load, verify, and clean metadata ───────────────
        cleaner  = DatasetCleaner(self.metadata_path, registry)
        raw_df   = cleaner.load_metadata()
        joined   = cleaner.verify_images_exist(raw_df)
        clean_df = cleaner.clean_dataset(joined)

        # ── Step 3: Split + OHE-encode localization ────────────────
        splitter = DatasetSplitter(
            train_size   = self.train_size,
            val_size     = self.val_size,
            test_size    = self.test_size,
            random_state = self.random_state
        )
        train_df, val_df, test_df = splitter.split(clean_df)

        # Expose the fitted encoder at the runner level for EffNetcnn.py
        self.loc_encoder = splitter.loc_encoder

        self._print_summary(train_df, val_df, test_df)
        return train_df, val_df, test_df

    def _print_summary(self,
                       train_df: pd.DataFrame,
                       val_df:   pd.DataFrame,
                       test_df:  pd.DataFrame) -> None:
        """Print final statistics and a usage guide for EffNetcnn.py."""
        _header("PIPELINE COMPLETE — SUMMARY")

        total = len(train_df) + len(val_df) + len(test_df)

        print(f"  {'Split':<8}  {'Images':>8}  {'Lesions':>8}  {'%':>6}")
        print(f"  {'-'*36}")
        for name, sdf in [("train", train_df),
                          ("val",   val_df),
                          ("test",  test_df)]:
            pct = len(sdf) / total * 100
            print(f"  {name:<8}  {len(sdf):>8,}  "
                  f"{sdf['lesion_id'].nunique():>8,}  {pct:>5.1f}%")
        print(f"  {'-'*36}")
        print(f"  {'TOTAL':<8}  {total:>8,}")

        print(f"\n  Columns in each split DataFrame:")
        print(f"    {list(train_df.columns)}")

        loc_cols = [c for c in train_df.columns if c.startswith("loc_")]
        print(f"\n  Localization OHE columns ({len(loc_cols)}):")
        print(f"    {loc_cols}")

        print(f"\n  Metadata feature vector for EfficientNetB3 dense branch:")
        meta_features = ["age", "sex"] + loc_cols
        print(f"    {meta_features}")
        print(f"    Total metadata features: {len(meta_features)}")

        print(f"\n  ── Usage in EffNetcnn.py ──────────────────────────────")
        print(f"  from images_cleaner import PipelineRunner")
        print(f"  runner = PipelineRunner(")
        print(f"      metadata_path = 'HAM10000_metadata_cleaned.csv',")
        print(f"      image_dirs    = ['HAM10000_images_part1',")
        print(f"                       'HAM10000_images_part2'],")
        print(f"  )")
        print(f"  train_df, val_df, test_df = runner.run()")
        print(f"  # runner.loc_encoder.categories_[0] → site name array")
        print()


# ═════════════════════════════════════════════════════════════════════
# ENTRY POINT (standalone run)
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── EDIT THESE PATHS BEFORE RUNNING ───────────────────────────
    METADATA_PATH = r"D:\study\year2\sem3\CSCI323project\HAM10000_metadata_cleaned.csv"
    IMAGE_DIR_1   = r"D:\study\year2\sem3\CSCI323project\chive\HAM10000_images_part_1"
    IMAGE_DIR_2   = r"D:\study\year2\sem3\CSCI323project\chive\HAM10000_images_part_2"

    DOWNLOAD_TO_PATH = r"D:\study\year2\sem3\CSCI323project\split_csv"

    runner = PipelineRunner(
        metadata_path = METADATA_PATH,
        image_dirs    = [IMAGE_DIR_1, IMAGE_DIR_2],
        train_size    = 0.70,
        val_size      = 0.15,
        test_size     = 0.15,
        random_state  = 42,
    )
    train_df, val_df, test_df = runner.run()

# ================================================================
# SAVE REPRODUCIBLE IMAGE SPLITS
# ================================================================
    try:
        os.makedirs(DOWNLOAD_TO_PATH, exist_ok=True)
        print(f"\nDirectory created or already exists: {DOWNLOAD_TO_PATH}")
        print("\nSaving split files:")

        train_df.to_csv(DOWNLOAD_TO_PATH + r"\train_split.csv", index=False)
        print(f" train_df saved as train_split.csv to {DOWNLOAD_TO_PATH}")

        val_df.to_csv(DOWNLOAD_TO_PATH + r"\val_split.csv", index=False)
        print(f" val_df saved as val_split.csv to {DOWNLOAD_TO_PATH}")

        test_df.to_csv(DOWNLOAD_TO_PATH + r"\test_split.csv", index=False)
        print(f" test_df saved as test_split.csv to {DOWNLOAD_TO_PATH}")

    except OSError as e:
        print(f"Error creating directory: {e}")

    

    
    
    
    