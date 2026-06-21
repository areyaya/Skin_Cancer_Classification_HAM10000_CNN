import pandas as pd
from pathlib import Path


FEATURE_DIR_REDU = Path(r"D:\study\year2\sem3\CSCI323project\effnetP2\reduced files")
FEATURE_DIR_META = Path(r"D:\study\year2\sem3\CSCI323project\effnetP2")


def verify_pair(split: str):

    feature_path = FEATURE_DIR_REDU / f"{split}_effnet_reduced.csv"
    meta_path = FEATURE_DIR_META / f"{split}_meta.csv"

    print(f"\nChecking {split.upper()}")

    # ------------------------------------------------
    # File existence
    # ------------------------------------------------
    assert feature_path.exists(), f"Missing file: {feature_path}"
    assert meta_path.exists(), f"Missing file: {meta_path}"

    # ------------------------------------------------
    # Load files
    # ------------------------------------------------
    feature_df = pd.read_csv(feature_path)
    meta_df = pd.read_csv(meta_path)


    # ------------------------------------------------
    # Duplicate checks (within each file)
    # ------------------------------------------------
    print("Checking duplicates...")

    feat_dupes = feature_df.duplicated().sum()
    meta_dupes = meta_df.duplicated().sum()

    assert feat_dupes == 0, f"{split}: Found {feat_dupes} duplicate rows in features"
    assert meta_dupes == 0, f"{split}: Found {meta_dupes} duplicate rows in metadata"

    if "image_id" in feature_df.columns:
        feat_id_dupes = feature_df["image_id"].duplicated().sum()
        assert feat_id_dupes == 0, f"{split}: Duplicate image_id in features ({feat_id_dupes})"

    if "image_id" in meta_df.columns:
        meta_id_dupes = meta_df["image_id"].duplicated().sum()
        assert meta_id_dupes == 0, f"{split}: Duplicate image_id in metadata ({meta_id_dupes})"

    # ------------------------------------------------
    # Missing value checks
    # ------------------------------------------------
    assert not feature_df.isnull().values.any(), (
        f"{split}: Missing values found in feature CSV"
    )

    ignore_cols = ["aug_params"]

    check_df = meta_df.drop(columns=ignore_cols, errors="ignore")

    assert not check_df.isnull().values.any(), (
    f"{split}: Missing values found in metadata CSV")

    # ------------------------------------------------
    # Optional image_id verification
    # ------------------------------------------------
    if "image_id" in feature_df.columns and "image_id" in meta_df.columns:

        assert feature_df["image_id"].equals(meta_df["image_id"]), (
            f"{split}: image_id ordering mismatch "
            f"between features and metadata"
        )

        print(" image_id ordering verified")

    print(
        f" VERIFIED\n"
        f"Features Shape : {feature_df.shape}\n"
        f"Metadata Shape : {meta_df.shape}"
    )


def check_cross_split_duplicates():

    print("\nChecking cross-split duplicates...")
    print("-" * 40)

    splits = ["train", "val", "test"]

    feature_ids = {}
    meta_ids = {}

    for split in splits:
        feat = pd.read_csv(FEATURE_DIR_REDU / f"{split}_effnet_reduced.csv")
        meta = pd.read_csv(FEATURE_DIR_META / f"{split}_meta.csv")

        feature_ids[split] = set(feat["image_id"].astype(str))
        meta_ids[split] = set(meta["image_id"].astype(str))

    # Feature leakage checks
    assert feature_ids["train"].isdisjoint(feature_ids["val"])
    assert feature_ids["train"].isdisjoint(feature_ids["test"])
    assert feature_ids["val"].isdisjoint(feature_ids["test"])

    # Metadata leakage checks
    assert meta_ids["train"].isdisjoint(meta_ids["val"])
    assert meta_ids["train"].isdisjoint(meta_ids["test"])
    assert meta_ids["val"].isdisjoint(meta_ids["test"])

    print("✓ No cross-split duplicates found")


def verify_all():

    print("=" * 60)
    print("VERIFYING FEATURE / METADATA FILES")
    print("=" * 60)

    verify_pair("train")
    verify_pair("val")
    verify_pair("test")

    check_cross_split_duplicates()

    print("\n" + "=" * 60)
    print("ALL FILES VERIFIED SUCCESSFULLY")
    print("=" * 60)


def create_multimodal_files():

    print("\n" + "=" * 60)
    print("CREATING MULTIMODAL FILES")
    print("=" * 60)

    for split in ["train", "val", "test"]:

        feature_path = FEATURE_DIR_REDU / f"{split}_effnet_reduced.csv"
        meta_path = FEATURE_DIR_META / f"{split}_meta.csv"

        feature_df = pd.read_csv(feature_path)
        meta_df = pd.read_csv(meta_path)

        # ------------------------------------------------
        # Merge on image_id
        # ------------------------------------------------
        merged_df = pd.merge(
            feature_df,
            meta_df,
            on="image_id",
            how="inner",
            validate="one_to_one"
        )

        # ------------------------------------------------
        # Verify row count preservation
        # ------------------------------------------------
        assert len(merged_df) == len(feature_df), (
            f"{split}: Merge changed row count "
            f"({len(feature_df)} -> {len(merged_df)})"
        )

        output_path = FEATURE_DIR_REDU / f"{split}_multimodal.csv"

        merged_df.to_csv(output_path, index=False)

        print(
            f" {split.upper()} saved\n"
            f"  Shape: {merged_df.shape}\n"
            f"  File : {output_path}"
        )

    print("\n✓ All multimodal files created successfully")

if __name__ == "__main__":

    verify_all()

    create_multimodal_files()

