"""
data_cleaner.py
===============
CSCI323 Modern Artificial Intelligence — Spring 2026

Dataset: HAM10000 Skin Lesion Metadata (HAM10000_metadata.csv)
          10,015 patient records, 7 columns

Purpose
-------
This file cleans the raw dataset and prepares it for downstream steps:
    1. Exploratory Data Analysis (EDA)
    2. Feature Engineering
    3. Model Training and Evaluation

Approach
--------
The cleaning pipeline follows a systematic, reasoning-first methodology:
    - Every decision is justified before it is executed.
    - No row is dropped without first checking whether it can be imputed.
    - Where multiple encoding strategies exist, the one that best preserves
      clinical meaning is chosen over the statistically convenient one.

Column reference
----------------
    lesion_id     — unique ID per lesion (multiple images may share one lesion)
    image_id      — unique ID per image (primary key)
    dx            — diagnosis label (7 classes) → one-hot encoded
    dx_type       — method of diagnosis (4 classes) → label encoded
    age           — patient age (float, 57 nulls) → median imputed
    sex           — patient sex (male/female/unknown) → binary + NaN for unknown
    localization  — body-site location (15 classes) → one-hot encoded

OOP Structure
-------------
    FileImporter  — loads the raw CSV from disk
    DataCleaner   — all cleaning logic (private methods + public pipeline)

The cleaned dataset is saved to:
    HAM10000_metadata_cleaned.csv
"""


# ---------- IMPORT LIBRARIES ----------
import shutil

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 1 — FileImporter
# ══════════════════════════════════════════════════════════════════════════════

class FileImporter:
    """
    Responsible solely for loading the raw CSV from disk.
    Keeping I/O separate from cleaning follows the Single Responsibility
    Principle: if the file location or format changes, only this class
    needs to be updated — the DataCleaner is unaffected.
    """

    def __init__(self, filepath: str):
        """
        Parameters
        ----------
        filepath : str
            Full path to the CSV file to be loaded.
        """
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        """
        Read the CSV into a DataFrame.

        Returns
        -------
        pd.DataFrame
            Raw, unmodified data exactly as it appears on disk.
        """
        df = pd.read_csv(self.filepath)

        # Replace any literal string "\N" (common in exported datasets) with
        # NaN so downstream logic treats it as a proper missing value.
        df.replace("\\N", np.nan, inplace=True)

        print(f"Loaded: {os.path.basename(self.filepath)}  ({len(df):,} rows × {df.shape[1]} columns)")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# CLASS 2 — DataCleaner
# ══════════════════════════════════════════════════════════════════════════════

class DataCleaner:
    """
    Executes the full data cleaning pipeline on the HAM10000 metadata.

    Public interface
    ----------------
        clean(df)  →  returns a fully cleaned DataFrame

    Private methods (called in order by clean)
    -------------------------------------------
        1. _fill_empty_cells            — standardise all missing values to NaN
        2. _remove_duplicates           — drop exact-duplicate rows   
        3. _encode_sex                  — binary encode sex column
        4. _impute_age                  — median imputation for the 57 null ages
        5. _label_encode_dx             — label encode diagnosis (dx)
        6. _one_hot_encode_localization — one-hot encode body-site location
    """

    def __init__(self):
        """
        Initialise storage for fitted transformers so they can be
        re-used during inference (i.e., transforming unseen data with
        the same fitted parameters as the training set).
        """
        self.label_encoders: dict = {}   # keyed by column name
        self.scaler = StandardScaler()   # fitted on numerical subset
        self.age_imputer = SimpleImputer(strategy="median")  # fitted on age

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC — main pipeline entry point
    # ──────────────────────────────────────────────────────────────────────────

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates the full cleaning pipeline by calling every private
        method in the correct dependency order and returns the finished
        DataFrame.

        Why a single orchestrator method?
        - It makes the sequence explicit and easy to audit.
        - Individual steps can be skipped or reordered without touching
          the step implementations themselves.

        Parameters
        ----------
        data : pd.DataFrame
            Raw DataFrame as returned by FileImporter.load().

        Returns
        -------
        pd.DataFrame
            Fully cleaned and encoded DataFrame ready for EDA / modelling.
        """
        df = data.copy()   # keep the original dataset intact for debugging/auditing

        print("\n" + "=" * 60)
        print("  DATA CLEANING PIPELINE — HAM10000 Metadata")
        print("=" * 60)
        print(f"  Starting shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

        df = self._fill_empty_cells(df)
        df = self._remove_duplicates(df)
        df = self._encode_sex(df)
        df = self._impute_age(df)
        #df = self._label_encode_dx_type(df) 
        #[CAN CAUSE DATA LEAKAGE - dx_type is a proxy for dx, so encoding it may leak information about the target variable into the features. This can lead to overfitting and poor generalization on unseen data.]

        df = self._label_encode_dx(df) #keep the target column to one column
        #df = self._one_hot_encode_localization(df)
        #df = self._standardize_age(df) #standardize only on train set 
        df = self._drop_id_columns(df)
        df = self._display_graphs(df)  # optional visualisation step to check distributions after cleaning
       

        print("\n" + "=" * 60)
        print(f"  Cleaning complete.")
        print(f"  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print("=" * 60 + "\n")
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1 — Standardise all missing representations to NaN
    # ──────────────────────────────────────────────────────────────────────────

    def _fill_empty_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace every variant of "empty" with np.nan so that all
        downstream pandas methods (isnull, dropna, fillna, SimpleImputer)
        work consistently.

        Source — regex approach adapted from:
            https://stackoverflow.com/a/21942746
            Posted by patricksurry, modified by community.
            Retrieved 2026-04-24, License CC BY-SA 4.0
        """
        # Regex ^\s*$ matches strings that are entirely whitespace (or empty).
        # Only applied to object (string) columns to avoid accidentally
        # converting numeric columns that contain a legitimate 0.
        df = df.replace(r'^\s*$', np.nan, regex=True)

        print("  [1] Standardised empty/whitespace strings → NaN.")
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2 — Remove duplicate rows
    # ──────────────────────────────────────────────────────────────────────────

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:  
        """
        Drop rows where every column value is identical to another row.

        Why remove duplicates?
        - Duplicate records inflate class frequencies and can bias model
          training by effectively up-weighting certain observations.
        - In a medical imaging dataset, a true duplicate (same image_id,
          same lesion_id, same diagnosis) almost certainly represents a
          data entry or export error rather than a real second patient.

        Why use drop_duplicates() with default keep='first'?
        - 'first' retains the earliest occurrence, which is a neutral
          choice when no timestamp or version column exists to prefer one
          copy over another.
        - The HAM10000 dataset does contain multiple images per lesion
          (same lesion_id, different image_id). These are NOT duplicates —
          drop_duplicates() considers all columns, so rows with distinct
          image_ids will never be removed.
        """
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)

        print(f"  [2] Removed {removed} exact duplicate rows. ({len(df):,} remain)")
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3 — Binary-encode the 'sex' column (FIXED - Dropped NaN sex values)
    # ──────────────────────────────────────────────────────────────────────────

    def _encode_sex(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map the 'sex' string column to a numeric binary column.

        Encoding scheme
        ---------------
            'female'  → 1
            'male'    → 0
            'unknown' → NaN  (treated as missing, not a third category)


        Why store the map as a class attribute?
        - If new data arrives at inference time, the same map can be applied
          consistently without refitting.
        """
        # Define the explicit mapping.
        sex_map = {
            'female':  1,
            'male':    0,
            'unknown': np.nan   # unknown → missing, not a third class
        }

        df['sex'] = df['sex'].map(sex_map)

        # Report how many unknowns were converted to NaN.
        n_unknown = df['sex'].isnull().sum()
        print(f"  [3] Binary-encoded 'sex'  (female=1, male=0). "
              f"{n_unknown} 'unknown' values set to NaN.")
        
        # Drop rows where column 'sex' is null ie. NaN
        drop_val = df[df['sex'].isnull()]
        df = df.drop(drop_val.index)
        print(f"  [3] Dropped {len(drop_val)} rows with missing 'sex' values.")
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4 — Impute missing age values
    # ──────────────────────────────────────────────────────────────────────────

    def _impute_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill the 57 missing values in the 'age' column using median imputation
        via sklearn's SimpleImputer.

        Why impute rather than drop?
        - Only 57 of 10,015 rows (0.57 %) have a missing age.
          Dropping them would waste otherwise complete, valuable records.
        - For clinical data, median imputation is preferred over mean
          imputation because age distributions are often slightly skewed
          (older patients tend to develop more lesions); the median is
          robust to this skew and to extreme values at either end.

        """
        null_before = df['age'].isnull().sum()

        # Reshape to 2-D array as required by sklearn transformers.
        age_array = df[['age']].values

        # fit_transform learns the median and immediately applies it.
        df['age'] = self.age_imputer.fit_transform(age_array)

        null_after = df['age'].isnull().sum()
        median_used = self.age_imputer.statistics_[0]

        print(f"  [4] Imputed {null_before} missing age values using median "
              f"({median_used:.1f} years). Nulls remaining: {null_after}.")
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5 — Label-encode 'dx_type'
    # ──────────────────────────────────────────────────────────────────────────
    # dx_type is a proxy for dx. Deciding whether to move to biopsy or not comes only after
    # initial clinical examination, thus encoding this column may leak information about the
    # target value, decreasing the model's ability to generalise on unseen data. Therefore,
    # this column will be dropped instead of being label encoded.
    #
    # Below is the code for label encoding kept for reference but commented out to
    # prevent accidental use.
    #
    # def _label_encode_dx_type(self, df: pd.DataFrame) -> pd.DataFrame:
    #     le = LabelEncoder()
    #     df['dx_type'] = le.fit_transform(df['dx_type'].astype(str))
    #     self.label_encoders['dx_type'] = le
    #     classes = dict(zip(le.classes_, le.transform(le.classes_)))
    #     print(f"  [5] Label-encoded 'dx_type'. Mapping: {classes}")
    #     return df


    # ──────────────────────────────────────────────────────────────────────────
    # STEP 6 — Label-encode 'dx' (diagnosis)
    # ──────────────────────────────────────────────────────────────────────────

    def _label_encode_dx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the 'dx' column from categorical string labels to integer codes 
        using label encoding. 

        ->  Why label encode 'dx'?
        To prevent the numerical values from meaning anything to the model,
        the output will be represented using softmax activation and cross-entropy loss
        which treats the output as categorical, thus preventing the model from 
        learning any ordinal relationships between the classes.

        This will be taken care of in the modelling stage, 
        thus the target column will be kept as one column and not one-hot encoded.

        There are a total of 7 diagnosis classes:
        - 'akiec' (Actinic keratoses and intraepithelial carcinoma / Bowen's disease)
        - 'bcc' (Basal cell carcinoma)
        - 'bkl' (Benign keratosis-like lesions)
        - 'df' (Dermatofibroma)
        - 'mel' (Melanoma)
        - 'nv' (Melanocytic nevi)
        - 'vasc' (Vascular lesions)
        """
        
        le = LabelEncoder()
        df['dx'] = le.fit_transform(df['dx'].astype(str))
        self.label_encoders['dx'] = le
        classes = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"  [6] Label-encoded 'dx'. Mapping: {classes}")
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 7 — One-hot encode 'localization'
    # ──────────────────────────────────────────────────────────────────────────
    # Commented out as this can be 

    #def _one_hot_encode_localization(self, df: pd.DataFrame) -> pd.DataFrame:
        #Convert the 'localization' column (body-site of the lesion) into
        #binary indicator columns using one-hot encoding.

        #Categories (15 body sites)
        
        #   abdomen, acral, back, chest, ear, face, foot, genital,
        #binary indicator columns using one-hot encoding.

        #Categories (15 body sites)
        #    abdomen, acral, back, chest, ear, face, foot, genital,
        #   hand, lower extremity, neck, scalp, trunk, unknown,
        #    upper extremity
        
        #dummies = pd.get_dummies(
        #    df['localization'], prefix='loc', drop_first=False, dtype=int
        #)
        #df = pd.concat([df.drop(columns=['localization']), dummies], axis=1)

        #new_cols = list(dummies.columns)
        #print(f"  [7] One-hot encoded 'localization' → {len(new_cols)} columns: {new_cols}")
        #return df


    # ──────────────────────────────────────────────────────────────────────────
    # STEP 8 — Drop non-informative ID columns
    # ──────────────────────────────────────────────────────────────────────────

    
    def _drop_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the 'dx_type' and 'image_id' columns from the cleaned
        DataFrame before modelling.
        """
        
        # dx_type is a proxy for dx, thus dropped to prevent data leakage
        id_cols = ['dx_type']
        try:  
            df = df.drop(columns=id_cols, errors='ignore')
            print(f"  [7] Dropped non-informative ID columns: {id_cols}")
            return df
        except KeyError as e:
            print(f"  [7] Error occurred while dropping ID columns: {e}")
        
        
    
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 9 — Plot graphs
    # ──────────────────────────────────────────────────────────────────────────
    def _display_graphs(self, df: pd.DataFrame):
        """
        Display graphs to visualize the distribution of the data.
        """
        # Plot age distribution
        plt.figure(figsize=(10, 6))

        #-------------------------------------------------------------
        # Plot age distribution
        sns.histplot(df['age'], bins=30, kde=True)
        plt.title('Age Distribution')
        plt.xlabel('Age (raw values)')
        plt.ylabel('Frequency')
        plt.show()

        #-------------------------------------------------------------
        # Plot sex distribution
        plot_df = df.copy()
        plot_df['sex'] = plot_df['sex'].map({
            0: 'Male',
            1: 'Female'
        })

        sns.countplot(
            data=df,
            x='dx',
            hue='sex'
        )
        plt.title('Disease Distribution by Sex')
        plt.xlabel('Diagnosis')
        plt.ylabel('Count')
        plt.show()

        #-------------------------------------------------------------
        # Plot localization distribution
        plot_df['dx'] = self.label_encoders['dx'].inverse_transform(plot_df['dx'])

        sns.countplot(x='localization', data=plot_df, hue='localization')
        plt.title('Lesion Localization')
        plt.xlabel('Localization')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        # Plot

        return df


# ──────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Calling main execution block of metadata_cleaner.py...")

    # ── FILE PATHS ─────────────────────────────────────────────
    RAW_DATA_PATH = r"D:\study\year2\sem3\CSCI323project\chive\HAM10000_metadata.csv"

    CLEANED_OUTPUT_PATH = r"D:\study\year2\sem3\CSCI323project\HAM10000_for_report_cleaned.csv"

    # ── 1. Load raw data ──────────────────────────────────────
    importer = FileImporter(RAW_DATA_PATH)
    raw_df = importer.load()

    # ── 2. Clean ──────────────────────────────────────────────
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean(raw_df)

    # ── 3. Preview ────────────────────────────────────────────
    print("\nCleaned DataFrame (first 5 rows):")
    print(cleaned_df.head().to_string())

    print("\nColumn dtypes after cleaning:")
    print(cleaned_df.dtypes)

    print("\nRemaining null counts:")
    print(cleaned_df.isnull().sum())

    # ── 4. Save cleaned CSV ──────────────────────────────────
    cleaned_df.to_csv(CLEANED_OUTPUT_PATH, index=False)

    print(f"\nCleaned dataset saved to:")
    print(CLEANED_OUTPUT_PATH)