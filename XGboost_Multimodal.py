"""
PartC_multimodal_classifier.py
===============================
CSCI323 Modern Artificial Intelligence — Spring 2026
University of Wollongong Dubai (UOWD)

Dataset : HAM10000 (7-class skin lesion classification)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Final multimodal skin lesion classifier combining:
    • Reduced EfficientNetB3 image features  (pre-computed, embedded in CSV)
    • Structured clinical metadata           (age, sex, localization OHE)

Classifier: XGBoost multiclass (softprob, GPU-accelerated if available)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POSITION IN THE FULL PROJECT PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    data_cleaner.py            complete — metadata cleaned, sex encoded,
                               age imputed, localization OHE fitted on train
    images_cleaner.py          complete — lesion-level stratified split,
                               ImageRegistry, DatasetSplitter
    EffNetcnn.py               complete — frozen EfficientNetB3 → 1536-dim
    Feature Reduction          complete — 1536 → reduced_dim (256 or 128),
                               merged with metadata into train/val/test
                               multimodal CSVs

    PartC_multimodal_classifier.py   ← THIS FILE (final step)

THIS FILE:
    1. Loads the pre-built train/val/test multimodal CSVs (image features +
       clinical metadata already merged, one row per lesion)
    2. Drops identifier / housekeeping columns not used as model input
    3. Splits each DataFrame into a feature matrix (X) and target array (y)
    4. Trains an XGBoost multiclass classifier (softprob, GPU if available)
    5. Evaluates on validation and test sets (full metric suite)
    6. Saves model, predictions, metrics, and diagnostic plots

THIS FILE MUST NEVER:
    • Re-run feature extraction or feature reduction
    • Fit any scaler, encoder, or imputer on val/test data
    • Modify the train/val/test split

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OOP STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    FeatureLoader        — loads the train/val/test multimodal CSVs and
                           returns them as DataFrames
    XGBoostClassifier    — builds, trains, and saves the XGBoost model;
                           handles class imbalance via sample_weight
    Evaluator            — full metric suite + confusion matrix + ROC plot
    MultimodalPipeline   — orchestrates all classes end-to-end, including
                           the column-dropping / X-y split logic

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from PartC_multimodal_classifier import MultimodalPipeline

    pipeline = MultimodalPipeline(
        train_path = r"D:\...\train_multimodal.csv",
        val_path   = r"D:\...\val_multimodal.csv",
        test_path  = r"D:\...\test_multimodal.csv",
        output_dir = r"D:\...\teamC_outputs",
    )
    pipeline.run()
"""

# ─────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────
import json
import os
import pickle
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120})

# ─────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────

# HAM10000 class map — matches data_cleaner.py integer encoding
DX_LABEL_MAP: Dict[int, str] = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc",
}
NUM_CLASSES: int = 7
CLASS_NAMES: List[str] = [DX_LABEL_MAP[i] for i in range(NUM_CLASSES)]

# Target column — isolated from each DataFrame to form y_train / y_val / y_test
LABEL_COL: str = "dx_y"

# Identifier / housekeeping columns present in the multimodal CSVs that are
# NOT model input features. These are dropped before the X / y split so
# they cannot leak into training as spurious numeric/categorical features.
DROP_COLS = [
    "file_path",
    "filepath",
    "aug_params",
    "image_id",
    "lesion_id",
    "dx_x",
]


# ─────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    """Print a visual section divider for readability in logs."""
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def _set_seed(seed: int = 42) -> None:
    """Seed Python random and NumPy for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


# ══════════════════════════════════════════════════════════════════════════
# CLASS 1 — FeatureLoader
# ══════════════════════════════════════════════════════════════════════════

class FeatureLoader:
    """
    Loads train, validation, and test multimodal CSVs (pre-computed image
    features + clinical metadata, already merged one row per lesion) and
    returns them as pandas DataFrames for downstream use by the pipeline.
    """

    def __init__(self, train_path: str, val_path: str, test_path: str) -> None:
        """
        Parameters
        ----------
        train_path : str
            Path to the training CSV file (e.g., 'train_multimodal.csv').
        val_path : str
            Path to the validation CSV file (e.g., 'val_multimodal.csv').
        test_path : str
            Path to the test CSV file (e.g., 'test_multimodal.csv').
        """
        _header("STEP 1 — Loading Multimodal Features & Metadata")

        # 1. Load the CSV files and save them as instance variables
        self.train_df = pd.read_csv(train_path)
        self.val_df   = pd.read_csv(val_path)
        self.test_df  = pd.read_csv(test_path)

        # 2. Display the DataFrame shapes in the output shell
        print(f"  Train DF shape : {self.train_df.shape}")
        print(f"  Val DF shape   : {self.val_df.shape}")
        print(f"  Test DF shape  : {self.test_df.shape}")
        print("  Data successfully loaded and saved to instance variables.\n")

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Return the loaded train, validation, and test DataFrames.

        Returns
        -------
        (train_df, val_df, test_df) : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        """
        return self.train_df, self.val_df, self.test_df

# ══════════════════════════════════════════════════════════════════════════
# CLASS 4 — XGBoostClassifier
# ══════════════════════════════════════════════════════════════════════════

class XGBoostClassifier:
    """
    XGBoost multiclass classifier with class imbalance handling.

    ─── HYPERPARAMETER CHOICES ───────────────────────────────────────────
        n_estimators   = 500   sufficient for convergence with early stopping
        max_depth      = 6     standard for tabular medical data; deeper
                               trees overfit on ~7,000 samples
        learning_rate  = 0.05  small enough for stable convergence with
                               early stopping; comparable to the XGBoost
                               paper defaults
        subsample      = 0.8   row subsampling reduces variance; standard
        colsample_bytree=0.8   column subsampling reduces variance
        min_child_weight=5     prevents splits on very small leaf nodes;
                               important with imbalanced classes
        gamma          = 1     minimum loss reduction for a split;
                               regularises against spurious splits
        reg_alpha      = 0.1   L1 regularisation (sparse feature selection)
        reg_lambda     = 1.0   L2 regularisation (default, weight decay)
        early_stopping = 30    stop if val mlogloss does not improve
                               for 30 rounds
    """

    def __init__(
        self,
        output_dir:      Path,
        n_estimators:    int   = 500,
        max_depth:       int   = 8,
        learning_rate:   float = 0.03,
        subsample:       float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int  = 5,
        gamma:           float = 1.0,
        reg_alpha:       float = 0.1,
        reg_lambda:      float = 1.0,
        early_stopping:  int   = 30,
        random_seed:     int   = 42,
    ) -> None:
        self.output_dir       = output_dir
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma            = gamma
        self.reg_alpha        = reg_alpha
        self.reg_lambda       = reg_lambda
        self.early_stopping   = early_stopping
        self.random_seed      = random_seed
        self.model: Optional[xgb.XGBClassifier] = None

    def _detect_device(self) -> str:
        """
        Detect whether GPU XGBoost is available.

        XGBoost's GPU support requires CUDA and the xgboost package
        compiled with GPU support. We detect availability by attempting
        a tiny GPU fit — if it raises an exception, we fall back to CPU.

        Returns
        -------
        "cuda" if GPU available, else "cpu"
        """
        try:
            probe = xgb.XGBClassifier(
                device="cuda", n_estimators=1, verbosity=0
            )
            probe.fit(
                np.zeros((10, 2), dtype=np.float32),
                np.zeros(10, dtype=np.int32),
                eval_set=[(np.zeros((2, 2), dtype=np.float32),
                           np.zeros(2, dtype=np.int32))],
                verbose=False,
            )
            print("  XGBoost device: CUDA (GPU)")
            return "cuda"
        except Exception:
            print("  XGBoost device: CPU (GPU not available or not supported)")
            return "cpu"

    def _compute_sample_weights(self, y_train: np.ndarray) -> np.ndarray:
        """
        Compute inverse-frequency sample weights for the training set.

        Uses sklearn's compute_sample_weight('balanced') which implements:
            weight(i) = total / (n_classes × count(class(i)))

        This is the correct imbalance correction for XGBoost multiclass:
        heavier weights on rare samples increase their gradient
        contribution during tree fitting.

        Parameters
        ----------
        y_train : np.ndarray (N,)  integer class labels 0-6

        Returns
        -------
        np.ndarray (N,)  float sample weights
        """
        weights = compute_sample_weight(class_weight="balanced", y=y_train)

        print("\n  Sample weights (class → weight):")
        for cls in range(NUM_CLASSES):
            mask = y_train == cls
            if mask.sum() > 0:
                w = weights[mask][0]
                print(f"    {DX_LABEL_MAP[cls]:8s}  count={mask.sum():,}  "
                      f"weight={w:.4f}")

        return weights

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> "XGBoostClassifier":
        """
        Train XGBoost with early stopping on the validation set.

        Validation set is used ONLY for early stopping — not for any
        hyperparameter search — to prevent data leakage.

        Parameters
        ----------
        X_train : (N_train, D)  combined feature matrix
        y_train : (N_train,)    integer labels
        X_val   : (N_val, D)    combined feature matrix (early stopping)
        y_val   : (N_val,)      integer labels

        Returns
        -------
        self
        """
        _header("STEP 4 — Training XGBoost Multiclass Classifier")

        device       = self._detect_device()
        sample_wts   = self._compute_sample_weights(y_train)

        print(f"\n  Training set : {X_train.shape}")
        print(f"  Val set      : {X_val.shape}")
        print(f"  n_estimators : {self.n_estimators}  "
              f"(with early stopping patience={self.early_stopping})")
        print(f"  max_depth    : {self.max_depth}")
        print(f"  learning_rate: {self.learning_rate}")

        self.model = xgb.XGBClassifier(
            # ── Task ───────────────────────────────────────────────────
            objective        = "multi:softprob",
            # softprob outputs calibrated class probabilities, required
            # for ROC-AUC computation and clinical deployment.

            num_class        = NUM_CLASSES,
            eval_metric      = "mlogloss",
            # mlogloss = multiclass log loss — the standard metric for
            # probabilistic multiclass classifiers; sensitive to
            # confidence as well as correctness.

            # ── Capacity / regularisation ──────────────────────────────
            n_estimators     = self.n_estimators,
            max_depth        = self.max_depth,
            learning_rate    = self.learning_rate,
            subsample        = self.subsample,
            colsample_bytree = self.colsample_bytree,
            min_child_weight = self.min_child_weight,
            gamma            = self.gamma,
            reg_alpha        = self.reg_alpha,
            reg_lambda       = self.reg_lambda,

            # ── Infrastructure ────────────────────────────────────────
            device           = device,
            random_state     = self.random_seed,
            n_jobs           = -1,
            verbosity        = 1,
            early_stopping_rounds = self.early_stopping,
        )

        self.model.fit(
            X_train,
            y_train,
            sample_weight   = sample_wts,
            eval_set        = [(X_val, y_val)],
            verbose         = 50,   # print every 50 rounds
        )

        best_round = self.model.best_iteration
        print(f"\n  Training complete — best round: {best_round}")

        # ── Save model ─────────────────────────────────────────────────
        model_path = self.output_dir / "xgboost_multimodal.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"  Model saved → {model_path}")

        # Also save in XGBoost native format for future XGB loading
        json_path = self.output_dir / "xgboost_multimodal.json"
        self.model.save_model(str(json_path))
        print(f"  Model (native) → {json_path}")

        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference and return predicted classes and probabilities.

        Parameters
        ----------
        X : (N, D)  combined feature matrix

        Returns
        -------
        preds : (N,)    predicted class indices
        probs : (N, 7)  softmax class probabilities
        """
        if self.model is None:
            raise RuntimeError("Call train() before predict().")

        probs = self.model.predict_proba(X)   # (N, 7) float
        preds = probs.argmax(axis=1)          # (N,)   int
        return preds, probs

    def plot_feature_importance(
        self,
        feature_names: List[str],
        top_n: int = 30,
    ) -> None:
        """
        Plot XGBoost feature importance (gain) for the top N features.

        'Gain' measures the average improvement in loss brought by a
        feature across all splits that use it — the most meaningful
        importance type for understanding which features drive predictions.

        Parameters
        ----------
        feature_names : ordered list matching columns of X_train
        top_n         : how many top features to display
        """
        if self.model is None:
            return

        importance = self.model.get_booster().get_score(importance_type="gain")
        # XGBoost names features f0, f1, f2, ... by default
        # Map them back to our feature names
        named_importance = {}
        for k, v in importance.items():
            idx = int(k[1:])   # strip "f" prefix → integer index
            if idx < len(feature_names):
                named_importance[feature_names[idx]] = v
            else:
                named_importance[k] = v

        sorted_imp = sorted(named_importance.items(),
                            key=lambda x: x[1], reverse=True)[:top_n]
        names, scores = zip(*sorted_imp) if sorted_imp else ([], [])

        fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, scores, align="center", color="steelblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Gain (average loss reduction per split)")
        ax.set_title(f"XGBoost Feature Importance (top {len(names)})\n"
                     "Higher = more predictive", fontweight="bold")
        plt.tight_layout()
        out = self.output_dir / "xgboost_feature_importance.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Feature importance plot → {out}")

# ══════════════════════════════════════════════════════════════════════════
# CLASS 6 — Evaluator
# ══════════════════════════════════════════════════════════════════════════

class Evaluator:
    """
    Computes the full metric suite and saves diagnostic plots and CSVs.

    ─── METRIC DESIGN RATIONALE ──────────────────────────────────────────
    Accuracy
        Reported for completeness. Misleading on imbalanced data —
        a model predicting only 'nv' scores ~67%, missing all
        clinically important malignant classes.

    Balanced Accuracy
        Mean per-class recall. Scale-invariant to class size.
        PRIMARY METRIC for the dissertation comparison table
        (XGBoost multimodal vs Teammate B image-only baseline).

    Macro Precision / Recall / F1
        Macro averaging gives equal weight to all 7 classes regardless
        of size. Critical for evaluating rare but clinically important
        classes: mel (melanoma), bcc, akiec.

    Multiclass ROC-AUC (OvR, macro)
        One-vs-Rest AUC for each class, then macro-averaged.
        Measures discrimination ability independent of classification
        threshold — important for clinical deployment where the threshold
        would be tuned per use case (e.g. high sensitivity for melanoma).
        Requires softmax probabilities from XGBoost softprob objective.

    Classification Report
        Per-class precision, recall, F1 — the core results table for
        the dissertation. Identifies which classes the model struggles
        with (expected: df and vasc are rare and visually similar to other
        classes).

    Confusion Matrix
        Normalised by true label (row-wise = recall per class).
        Reveals class-pair confusions: mel/nv and bkl/nv are the most
        clinically significant confusion pairs in HAM10000.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def compute_metrics(
        self,
        preds:      np.ndarray,
        labels:     np.ndarray,
        probs:      np.ndarray,
        split:      str,
        model_name: str = "XGBoost",
    ) -> Dict[str, float]:
        """
        Compute all metrics, print results, and return as a dict.

        Parameters
        ----------
        preds      : (N,)   predicted class indices
        labels     : (N,)   ground-truth class indices
        probs      : (N,7)  class probabilities
        split      : "val" | "test"
        model_name : label for logging

        Returns
        -------
        dict  metric_name → float
        """
        _header(f"EVALUATION — {model_name} — {split.upper()}")

        acc      = accuracy_score(labels, preds)
        bal_acc  = balanced_accuracy_score(labels, preds)
        prec     = precision_score(labels, preds, average="macro",
                                   zero_division=0)
        rec      = recall_score(labels, preds, average="macro",
                                zero_division=0)
        f1       = f1_score(labels, preds, average="macro", zero_division=0)

        # ROC-AUC: binarise for one-vs-rest multiclass AUC
        lb = label_binarize(labels, classes=list(range(NUM_CLASSES)))
        try:
            auc = roc_auc_score(lb, probs, average="macro",
                                multi_class="ovr")
        except ValueError as e:
            print(f"  WARNING: ROC-AUC could not be computed: {e}")
            auc = float("nan")

        metrics = {
            "model":             model_name,
            "split":             split,
            "accuracy":          round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "precision_macro":   round(prec, 4),
            "recall_macro":      round(rec, 4),
            "f1_macro":          round(f1, 4),
            "roc_auc_macro":     round(auc, 4),
        }

        print(f"\n  Accuracy          : {acc:.4f}")
        print(f"  Balanced Accuracy : {bal_acc:.4f}  ← primary metric")
        print(f"  Precision (macro) : {prec:.4f}")
        print(f"  Recall    (macro) : {rec:.4f}")
        print(f"  F1        (macro) : {f1:.4f}")
        print(f"  ROC-AUC   (macro) : {auc:.4f}")
        print(f"\n  Classification Report:\n")
        print(classification_report(
            labels, preds,
            target_names=CLASS_NAMES,
            zero_division=0,
        ))

        return metrics

    def plot_confusion_matrix(
        self,
        labels:     np.ndarray,
        preds:      np.ndarray,
        split:      str,
        model_name: str = "XGBoost",
    ) -> None:
        """
        Plot and save the normalised confusion matrix (row = recall).

        Normalisation by true label (row-wise) shows recall per class,
        making it visually apparent which classes are most confused
        regardless of their size. Raw counts would make the 'nv' row
        dominate the colour scale.
        """
        cm = confusion_matrix(labels, preds, normalize="true")

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot        = True,
            fmt          = ".2f",
            xticklabels  = CLASS_NAMES,
            yticklabels  = CLASS_NAMES,
            cmap         = "Blues",
            vmin         = 0,
            vmax         = 1,
            ax           = ax,
            linewidths   = 0.5,
        )
        ax.set_title(
            f"Confusion Matrix — {model_name} — {split.upper()}\n"
            f"(normalised by true label = recall per row)",
            fontweight="bold",
        )
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        plt.tight_layout()

        fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}_{split}.png"
        out   = self.output_dir / fname
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Confusion matrix → {out}")

    def plot_roc_curves(
        self,
        labels:     np.ndarray,
        probs:      np.ndarray,
        split:      str,
        model_name: str = "XGBoost",
    ) -> None:
        """
        Plot per-class ROC curves (one-vs-rest) on a single figure.

        Shows the trade-off between true positive rate and false positive
        rate for each class independently. Melanoma (mel) and basal cell
        carcinoma (bcc) ROC curves are most clinically relevant.
        """
        from sklearn.metrics import roc_curve

        lb     = label_binarize(labels, classes=list(range(NUM_CLASSES)))
        fig, ax = plt.subplots(figsize=(9, 7))

        for cls in range(NUM_CLASSES):
            try:
                fpr, tpr, _ = roc_curve(lb[:, cls], probs[:, cls])
                auc_val     = roc_auc_score(lb[:, cls], probs[:, cls])
                ax.plot(fpr, tpr,
                        label=f"{CLASS_NAMES[cls]} (AUC={auc_val:.2f})",
                        lw=1.8)
            except ValueError:
                pass

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(
            f"ROC Curves (one-vs-rest) — {model_name} — {split.upper()}",
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()

        fname = f"roc_curves_{model_name.lower().replace(' ', '_')}_{split}.png"
        out   = self.output_dir / fname
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  ROC curves → {out}")

    def save_predictions(
        self,
        labels:     np.ndarray,
        preds:      np.ndarray,
        probs:      np.ndarray,
        split:      str,
        model_name: str = "XGBoost",
    ) -> None:
        """
        Save per-sample predictions and probabilities to CSV.

        The CSV contains one row per test/val sample with:
            true_label, true_class_name, predicted_label,
            predicted_class_name, prob_akiec, ..., prob_vasc
        Useful for downstream error analysis and dissertation appendix.
        """
        rows = []
        for i in range(len(labels)):
            row = {
                "true_label":          int(labels[i]),
                "true_class":          DX_LABEL_MAP[int(labels[i])],
                "predicted_label":     int(preds[i]),
                "predicted_class":     DX_LABEL_MAP[int(preds[i])],
                "correct":             int(labels[i]) == int(preds[i]),
            }
            for cls in range(NUM_CLASSES):
                row[f"prob_{CLASS_NAMES[cls]}"] = round(float(probs[i, cls]), 5)
            rows.append(row)

        df   = pd.DataFrame(rows)
        fname = f"predictions_{model_name.lower().replace(' ', '_')}_{split}.csv"
        out   = self.output_dir / fname
        df.to_csv(out, index=False)
        print(f"  Predictions CSV → {out}  ({len(df)} rows)")



# ══════════════════════════════════════════════════════════════════════════
# CLASS 7 — MultimodalPipeline
# ══════════════════════════════════════════════════════════════════════════

class MultimodalPipeline:
    """
    Orchestrates the full Team C multimodal classification workflow.

    Execution sequence
    ──────────────────
        Step 1  Load train/val/test multimodal CSVs (FeatureLoader)
        Step 2  Drop identifier/housekeeping columns, then split each
                DataFrame into a feature matrix (X) and target array (y)
        Step 3  Train XGBoost with class-balanced sample weights
        Step 4  Evaluate on val and test sets
        Step 5  Save metrics summary, predictions, plots

    No data leakage guarantee
    ─────────────────────────
        • Column dropping and the X/y split are applied identically and
          independently to train, val, and test — no fitting happens here
        • XGBoost early stopping uses val set for round selection only
          (no hyperparameter search on val)
        • sample_weight is computed from y_train only
        • Val and test sets are never touched during training
    """

    def __init__(
        self,
        train_path:      str,
        val_path:        str,
        test_path:       str,
        output_dir:      str,
        random_seed:     int   = 42,
        # XGBoost hyperparameters (can be overridden)
        n_estimators:    int   = 500,
        max_depth:       int   = 8,
        learning_rate:   float = 0.03,
        subsample:       float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int  = 5,
        gamma:           float = 1.0,
        reg_alpha:       float = 0.1,
        reg_lambda:      float = 1.0,
        early_stopping:  int   = 30,
    ) -> None:
        self.train_path  = train_path
        self.val_path    = val_path
        self.test_path   = test_path
        self.output_dir  = Path(output_dir)
        self.random_seed = random_seed

        # XGBoost config
        self.xgb_kwargs = dict(
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            learning_rate    = learning_rate,
            subsample        = subsample,
            colsample_bytree = colsample_bytree,
            min_child_weight = min_child_weight,
            gamma            = gamma,
            reg_alpha        = reg_alpha,
            reg_lambda       = reg_lambda,
            early_stopping   = early_stopping,
            random_seed      = random_seed,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _split_features_and_target(
        df: pd.DataFrame,
        split_name: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Drop identifier/housekeeping columns, isolate the target column,
        and return the remaining columns as a numpy feature matrix.

        Parameters
        ----------
        df         : raw multimodal DataFrame for one split (train/val/test)
        split_name : "train" | "val" | "test"  (used for logging only)

        Returns
        -------
        X            : (N, D)  float32 feature matrix (all columns except
                       DROP_COLS and LABEL_COL)
        y            : (N,)    int32 target array (the 'dx' column)
        feature_names: ordered list of column names corresponding to X
        """
        # Drop identifier / housekeeping columns that are not model input.
        # errors="ignore" so a split missing one of these columns doesn't crash.
        cleaned = df.drop(columns=DROP_COLS, errors="ignore")

        # Isolate the target column.
        y = cleaned[LABEL_COL].to_numpy(dtype=np.int32)

        # Everything else (after dropping identifiers and the target)
        # forms the feature matrix.
        X_df = cleaned.drop(columns=[LABEL_COL])
        X = X_df.to_numpy(dtype=np.float32)

        print(f"  [{split_name}] X shape: {X.shape}  y shape: {y.shape}")

        return X, y, list(X_df.columns)

    def run(self) -> None:
        """Execute the full multimodal classification pipeline."""
        _header("HAM10000 — TEAM C MULTIMODAL CLASSIFIER")
        print(f"  Train CSV    : {self.train_path}")
        print(f"  Val CSV      : {self.val_path}")
        print(f"  Test CSV     : {self.test_path}")
        print(f"  Output dir   : {self.output_dir}")
        print(f"  Random seed  : {self.random_seed}")

        _set_seed(self.random_seed)

        # ── Step 1: Load train/val/test multimodal CSVs ────────────────
        loader = FeatureLoader(self.train_path, self.val_path, self.test_path)
        train_df, val_df, test_df = loader.load_all()

        # ── Step 2: Drop identifier columns, split into X / y ───────────
        _header("STEP 2 — Dropping Identifier Columns & Splitting X / y")
        print(f"  Dropping columns: {DROP_COLS}")
        X_train, y_train, feature_names = self._split_features_and_target(
            train_df, "train"
        )
        X_val, y_val, _ = self._split_features_and_target(val_df, "val")
        X_test, y_test, _ = self._split_features_and_target(test_df, "test")

        print(f"\n  Label distributions:")
        for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            dist = {DX_LABEL_MAP[c]: int((y == c).sum())
                    for c in range(NUM_CLASSES)}
            print(f"  {name}: {dist}")

        print(f"\n  Final feature vector dimension: {X_train.shape[1]}")

        # ── Step 3: Train XGBoost ───────────────────────────────────────
        xgb_clf = XGBoostClassifier(
            output_dir = self.output_dir,
            **self.xgb_kwargs,
        )
        xgb_clf.train(X_train, y_train, X_val, y_val)

        # ── Step 4: Evaluate XGBoost ─────────────────────────────────────
        evaluator    = Evaluator(self.output_dir)
        all_metrics: List[Dict] = []

        for split, X, y in [("val",  X_val,  y_val),
                             ("test", X_test, y_test)]:
            preds, probs = xgb_clf.predict(X)
            metrics = evaluator.compute_metrics(
                preds, y, probs, split, "XGBoost"
            )
            all_metrics.append(metrics)
            evaluator.plot_confusion_matrix(y, preds, split, "XGBoost")
            evaluator.plot_roc_curves(y, probs, split, "XGBoost")
            evaluator.save_predictions(y, preds, probs, split, "XGBoost")

        # Feature importance (top 30)
        xgb_clf.plot_feature_importance(feature_names, top_n=30)

        # ── Step 5: Save combined metrics summary ───────────────────────
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = self.output_dir / "metrics_summary.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n  Metrics summary → {metrics_path}")

        self._print_final_summary(metrics_df)

        _header("PIPELINE COMPLETE")
        print(f"  All outputs saved to: {self.output_dir}/")
        print(f"\n  Key files:")
        print(f"    xgboost_multimodal.json          ← trained model")
        print(f"    metrics_summary.csv              ← all metrics")
        print(f"    predictions_xgboost_test.csv     ← per-sample predictions")
        print(f"    confusion_matrix_xgboost_test.png")
        print(f"    roc_curves_xgboost_test.png")
        print(f"    xgboost_feature_importance.png")

    @staticmethod
    def _print_final_summary(metrics_df: pd.DataFrame) -> None:
        """Print a clean comparison table of all model / split combinations."""
        _header("RESULTS SUMMARY")
        cols = ["model", "split", "accuracy", "balanced_accuracy",
                "f1_macro", "roc_auc_macro"]
        available = [c for c in cols if c in metrics_df.columns]
        print(metrics_df[available].to_string(index=False))
        print()
        print("  Primary metric for dissertation: balanced_accuracy")
        print("  Compare XGBoost (multimodal) vs Teammate B (image-only CNN)")


# ─────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  PartC_multimodal_classifier.py")
    print("  HAM10000 Multimodal XGBoost Classifier")
    print("  CSCI323 Modern Artificial Intelligence — Spring 2026, UOWD")
    print("=" * 68)

    # ── FILE PATHS — update to match your local directory ─────────────
    TRAIN_PATH = r"D:\study\year2\sem3\CSCI323project\effnetP2\reduced files\train_multimodal.csv"
    VAL_PATH   = r"D:\study\year2\sem3\CSCI323project\effnetP2\reduced files\val_multimodal.csv"
    TEST_PATH  = r"D:\study\year2\sem3\CSCI323project\effnetP2\reduced files\test_multimodal.csv"
    OUTPUT_DIR = r"D:\study\year2\sem3\CSCI323project\effnetP2\reduced files\xgboost_multimodal_outputs_v2"

    pipeline = MultimodalPipeline(
        train_path       = TRAIN_PATH,
        val_path         = VAL_PATH,
        test_path        = TEST_PATH,
        output_dir       = OUTPUT_DIR,
        random_seed      = 42,

        # XGBoost hyperparameters
        n_estimators     = 500,
        max_depth        = 8,
        learning_rate    = 0.03,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        gamma            = 1.0,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        early_stopping   = 30,
    )
    pipeline.run()