# =============================================================================
# tsne_analytics.py
# HAM10000 Skin Lesion Classification — t-SNE Data Analytics
# CSCI323 Modern Artificial Intelligence | Spring 2026 | UOWD
# =============================================================================
#
#   CHECKPOINT A  (EfficientNetB3 raw features)
#       SOURCE = "effnet"
#       FEATURES_PATH → train_features.npy
#       LABELS_PATH   → train_meta.csv
#
#   CHECKPOINT C  (Custom CNN raw features)
#       SOURCE = "cnn"
#       FEATURES_PATH → train_features.npy  
#       LABELS_PATH   → train_labels.npy  
#
# OUTPUT:
#   Panel 1 — all 7 classes plotted by colour
#   Panel 2 — minority classes highlighted, nv (dominant class) greyed out


# SECTION 0 — IMPORTS
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")


#SECTION 1 — CONFIG
#"effnet" for Checkpoint A  (raw EfficientNetB3 features, .npy + meta.csv)
#"cnn" for Checkpoint C  (raw traditional CNN features, .npy + labels.npy)
SOURCE = "effnet"                                            

# Path to your FEATURE file
FEATURES_PATH = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\Opus EffNet Extracted\train_features.npy"

# Path to your LABELS file
LABELS_PATH   = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\Opus EffNet Extracted\train_meta.csv"

# Folder where the output 
OUTPUT_DIR    = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\Opus EffNet - NN Output"          

# Title that appears at the top of the plot
PLOT_TITLE    = "t-SNE of Extracted EfficientNetB3 Features"      

# Name of the saved .png file
OUTPUT_FILE   = "tsne_effnet_train.png"            


#SECTION 2 — CLASS LABELS AND COLOURS 
# dx class index
CLASS_NAMES = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc",
}
NUM_CLASSES = 7

CLASS_COLORS = [
    "#E69F00",   # 0 akiec  — orange
    "#56B4E9",   # 1 bcc    — sky blue
    "#009E73",   # 2 bkl    — teal green
    "#F0E442",   # 3 df     — yellow
    "#D55E00",   # 4 mel    — vermillion  (clinically most important class)
    "#0072B2",   # 5 nv     — deep blue
    "#CC79A7",   # 6 vasc   — pink/purple
]

NV_CLASS = 5   # nv is the dominant class


#SECTION 3 — t-SNE PARAMETERS
#PERPLEXITY: Controls how many "neighbours" each point pays attention to.
#   Range: 5–50. With ~7000–8000 training images, 40 is well-suited.
#   Higher means smoother but blurred clusters. Lower means fragmented micro-clusters.
#
# N_ITER: How many optimisation steps t-SNE runs. 1000 is the standard minimum for convergence. More = slower but cleaner.

PERPLEXITY   = 40
N_ITER       = 1000
RANDOM_SEED  = 42


# SECTION 4 — LOAD FEATURES AND LABELS
print("\n" + "="*50)
print(f" CHECKPOINT: {SOURCE.upper()}  |  t-SNE DATA ANALYTICS")
print("="*50)

print(f"\n[LOAD] Source type : {SOURCE}")
print(f"[LOAD] Features : {FEATURES_PATH}")

if SOURCE == "effnet":
    # Checkpoint A: raw EfficientNetB3 features
    X     = np.load(FEATURES_PATH).astype(np.float32)
    meta  = pd.read_csv(LABELS_PATH)
    y     = meta["dx"].values.astype(int)
    print(f"[LOAD] Labels (csv): {LABELS_PATH}")

elif SOURCE == "cnn":
    # Checkpoint C: raw traditional CNN features
    X     = np.load(FEATURES_PATH).astype(np.float32)
    y_raw = np.load(LABELS_PATH)

    # if labels are one-hot encoded shape (N, 7), convert to integer indices
    if y_raw.ndim == 2:
        y = np.argmax(y_raw, axis=1).astype(int)
    else:
        y = y_raw.astype(int)
    print(f"[LOAD] Labels (npy): {LABELS_PATH}")

else:
    raise ValueError(f"SOURCE must be 'effnet' or 'cnn' — got '{SOURCE}'")

# Alignment check: features and labels must have the same number of rows
assert X.shape[0] == y.shape[0], (f"MISMATCH: features have {X.shape[0]} rows but labels have {y.shape[0]} rows.\n")

print(f"\n[LOAD] Feature matrix : {X.shape} | ({X.shape[0]} images × {X.shape[1]} features per image)")
print(f"[LOAD] Label vector : {y.shape}")
print(f"[LOAD] Alignment check : PASSED")

# Count samples per class 
class_counts = {c: int((y == c).sum()) for c in range(NUM_CLASSES)}
print("\n[LOAD] Class distribution:")
for c, name in CLASS_NAMES.items():
    bar = "█" * (class_counts[c] // 100)
    print(f"  {c} {name:7s}  n={class_counts[c]:5d}  {bar}")


# SECTION 5 — RUN t-SNE
print(f"\n[TSNE] Running t-SNE — {X.shape[0]:,} samples × {X.shape[1]} dims")
print(f"[TSNE] perplexity={PERPLEXITY}  n_iter={N_ITER}  init=pca  seed={RANDOM_SEED}")

tsne = TSNE(
    n_components  = 2,
    perplexity    = PERPLEXITY,
    max_iter      = N_ITER,       # renamed from n_iter 
    learning_rate = "auto",       
    init          = "pca",        # PCA init = more stable, reproducible starting point than random
    metric        = "euclidean",
    random_state  = RANDOM_SEED,
)
embedding = tsne.fit_transform(X)    # shape: (N, 2)
print(f"[TSNE] Embedding complete — shape: {embedding.shape}")

# Save embedding
os.makedirs(OUTPUT_DIR, exist_ok=True)
embed_save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE.replace(".png", "_embedding.npy"))
np.save(embed_save_path, embedding)
print(f"[TSNE] Embedding saved to: {embed_save_path}")
print(f"      reload later with: embedding = np.load(r'{embed_save_path}')")


#SECTION 6 — TWO-PANEL DARK PLOT
# PANEL 1 (left)  — all 7 classes in colour
# PANEL 2 (right) — nv (dominant class) greyed into background, minority classes drawn in colour on top
# Sort classes by count descending

plot_order = sorted(class_counts.keys(), key=lambda c: class_counts[c], reverse=True)

# Dark background colours
FIG_BG  = "#1a1a2e"   # figure background
AX_BG   = "#0d0d1a"   # axes background
TXT_COL = "white"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(FIG_BG)
fig.suptitle(
    f"{PLOT_TITLE}\n"
    f"HAM10000 — {X.shape[0]:,} images | {X.shape[1]} input dims"
    f"perplexity={PERPLEXITY}",
    fontsize=13, fontweight="bold", color=TXT_COL, y=1.01
)

for ax in (ax1, ax2):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TXT_COL, labelsize=8)
    ax.set_xlabel("t-SNE Dimension 1", color=TXT_COL, fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", color=TXT_COL, fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

# --------------------------
# PANEL 1: all 7 classes
# --------------------------
ax1.set_title("All 7 Classes", color=TXT_COL, fontsize=12, pad=10)
legend_handles_p1 = []

for cls in plot_order:
    mask = (y == cls)
    ax1.scatter(
        embedding[mask, 0], embedding[mask, 1],
        c     = CLASS_COLORS[cls],
        s     = 8,
        alpha = 0.65,
        linewidths = 0,
    )
    patch = mpatches.Patch(
        color = CLASS_COLORS[cls],
        label = f"{CLASS_NAMES[cls]}   n={class_counts[cls]:,}"
    )
    legend_handles_p1.append(patch)

leg1 = ax1.legend(
    handles    = legend_handles_p1,
    loc        = "upper right",
    fontsize   = 9,
    framealpha = 0.25,
    facecolor  = "#222244",
    edgecolor  = "#555555",
    labelcolor = TXT_COL,
    title      = "dx class",
    title_fontsize = 9,
)
leg1.get_title().set_color(TXT_COL)

# ------------------------------------------------------------------
# PANEL 2: minority classes highlighted — nv greyed into background
# ------------------------------------------------------------------
ax2.set_title("Minority Classes (nv greyed)", color=TXT_COL, fontsize=12, pad=10)
legend_handles_p2 = []

# Draw nv first as grey background layer
nv_mask = (y == NV_CLASS)
ax2.scatter(
    embedding[nv_mask, 0], embedding[nv_mask, 1],
    c          = "#404040",
    s          = 5,
    alpha      = 0.18,
    linewidths = 0,
)
legend_handles_p2.append(mpatches.Patch(
    color = "#404040",
    label = f"nv (background)   n={class_counts[NV_CLASS]:,}"
))

# Draw minority classes on top 
minority_order = [c for c in plot_order if c != NV_CLASS]
for cls in minority_order:
    mask = (y == cls)
    ax2.scatter(
        embedding[mask, 0], embedding[mask, 1],
        c          = CLASS_COLORS[cls],
        s          = 14,
        alpha      = 0.85,
        linewidths = 0,
    )
    legend_handles_p2.append(mpatches.Patch(
        color = CLASS_COLORS[cls],
        label = f"{CLASS_NAMES[cls]}   n={class_counts[cls]:,}"
    ))

leg2 = ax2.legend(
    handles    = legend_handles_p2,
    loc        = "upper right",
    fontsize   = 9,
    framealpha = 0.25,
    facecolor  = "#222244",
    edgecolor  = "#555555",
    labelcolor = TXT_COL,
    title      = "dx class",
    title_fontsize = 9,
)
leg2.get_title().set_color(TXT_COL)

plt.tight_layout()

# Save and show
save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
print(f"[PLOT] Saved: {save_path}")
plt.show()
