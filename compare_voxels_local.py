import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.processing import resample_to_output, resample_from_to
from pathlib import Path

print("Comparing local PET prep outputs")
arun="run04"
brun="run09"

a_path = f"/home/roland/uni/thesisprep/ds001420-download/derivatives/petprep/{arun}/sub-dasb01/ses-baseline/pet/sub-dasb01_ses-baseline_space-T1w_desc-preproc_pet.nii.gz"
b_path = f"/home/roland/uni/thesisprep/ds001420-download/derivatives/petprep/{brun}/sub-dasb01/ses-baseline/pet/sub-dasb01_ses-baseline_space-T1w_desc-preproc_pet.nii.gz"

mask1_path = f"/home/roland/uni/thesisprep/ds001420-download/derivatives/petprep/{arun}/sub-dasb01/ses-baseline/pet/sub-dasb01_ses-baseline_space-T1w_desc-brain_mask.nii.gz"
mask2_path = f"/home/roland/uni/thesisprep/ds001420-download/derivatives/petprep/{brun}/sub-dasb01/ses-baseline/pet/sub-dasb01_ses-baseline_space-T1w_desc-brain_mask.nii.gz"

a = nib.load(a_path)
b = nib.load(b_path)
m1 = nib.load(mask1_path)
m2 = nib.load(mask2_path)

A_raw = a.get_fdata(dtype=np.float32)
B_raw = b.get_fdata(dtype=np.float32)

# Use copies for averaged workflow, keep raw for per-frame plots
A = A_raw.copy()
B = B_raw.copy()

if A.ndim == 4:
    print("A is 4D, averaging across frames")
    A = A.mean(axis=-1)

if B.ndim == 4:
    print("B is 4D, averaging across frames")
    B = B.mean(axis=-1)


M1 = m1.get_fdata(dtype=np.float32) > 0

# Alternative: DOES NOT WORK YET: resample B to A's grid if shapes differ
if A_raw.shape[:3] != B_raw.shape[:3]:
    print("Resampling using nibabel...")
    
    # Resample B to match A's voxel grid
    b_resampled = resample_from_to(b, a)
    B_raw = b_resampled.get_fdata(dtype=np.float32)
    
    # Also resample mask2
    m2_resampled_obj = resample_from_to(m2, a)
    m2_resampled = m2_resampled_obj.get_fdata(dtype=np.float32) > 0
    M2 = m2_resampled
else:
    M2 = m2.get_fdata(dtype=np.float32) > 0

print("same shape:", A.shape == B.shape)
print("same affine:", np.allclose(a.affine, b.affine))

diff = A - B
absdiff = np.abs(diff)

# intersection brain mask and exclude near zero background
M = M1 & M2 & (A > 0.01) & (B > 0.01)

vals = absdiff[M].ravel()

print("mean abs diff in mask:", vals.mean())
print("max abs diff in mask:", vals.max())
print("p95 abs diff in mask:", np.quantile(vals, 0.95))

tag = f"{arun}_vs_{brun}"

# Create subdirectories
data_absdiff_dir = Path("./data/absdiff")
data_normalized_dir = Path("./data/normalized")
plots_boxplot_dir = Path("./plots/boxplot")
plots_table_dir = Path("./plots/table")

data_absdiff_dir.mkdir(parents=True, exist_ok=True)
data_normalized_dir.mkdir(parents=True, exist_ok=True)
plots_boxplot_dir.mkdir(parents=True, exist_ok=True)
plots_table_dir.mkdir(parents=True, exist_ok=True)

absdiff_path = data_absdiff_dir / f"{tag}.nii.gz"
normalized_path = data_normalized_dir / f"{tag}.nii.gz"
boxplot_path = plots_boxplot_dir / f"abs_{tag}.png"
perframe_boxplot_path = plots_boxplot_dir / f"perframe_{tag}.png"
percent_boxplot_path = plots_boxplot_dir / f"perframe_percent_{tag}.png"
perframe_table_path = plots_table_dir / f"abs_{tag}.png"
perframe_percent_table_path = plots_table_dir / f"percent_{tag}.png"

# Apply brain mask when saving - only keep data within the brain
absdiff_masked = absdiff.copy()
absdiff_masked[~M] = 0

out = nib.Nifti1Image(absdiff_masked.astype(np.float32), a.affine, a.header)
nib.save(out, absdiff_path)

# Calculate normalized (percent) differences
normalized = np.zeros_like(A)
with np.errstate(divide='ignore', invalid='ignore'):
    normalized = (absdiff / A) * 100.0
    normalized[~np.isfinite(normalized)] = 0  # Handle division by zero

# Apply brain mask to normalized data as well
normalized[~M] = 0

out_normalized = nib.Nifti1Image(normalized.astype(np.float32), a.affine, a.header)
nib.save(out_normalized, normalized_path)

plt.figure()
plt.boxplot([vals], showfliers=False)
plt.xticks([1], ["abs diff in common brain mask"])
plt.ylabel("Absolute difference")
plt.title(f"PET absolute differences {arun} vs {brun}")
plt.savefig(boxplot_path, dpi=150)
plt.close()

print("Saved", absdiff_path)
print("Saved", normalized_path)
print("Saved", boxplot_path)

# New: per-frame boxplot without averaging
if A_raw.ndim == 4 and B_raw.ndim == 4:
    n_frames = A_raw.shape[-1]
    perframe_vals = []
    for i in range(n_frames):
        diff_i = A_raw[..., i] - B_raw[..., i]
        absdiff_i = np.abs(diff_i)
        mask_i = M1 & M2 & (A_raw[..., i] > 0.01) & (B_raw[..., i] > 0.01)
        vals_i = absdiff_i[mask_i].ravel()
        perframe_vals.append(vals_i)

    plt.figure(figsize=(max(6, n_frames * 0.6), 4))
    plt.boxplot(perframe_vals, showfliers=False)
    plt.xticks(list(range(1, n_frames + 1)), [f"t{i}" for i in range(n_frames)], rotation=45)
    plt.ylabel("Absolute difference")
    plt.title(f"PET absolute differences per frame {arun} vs {brun}")
    plt.tight_layout()
    plt.savefig(perframe_boxplot_path, dpi=150)
    plt.close()
    print("Saved", perframe_boxplot_path)

    # Table for per-frame absolute differences
    table_data_abs = []
    for i, vals_i in enumerate(perframe_vals):
        if len(vals_i) > 0:
            min_val = vals_i.min()
            q1_val = np.quantile(vals_i, 0.25)
            median_val = np.quantile(vals_i, 0.5)
            q3_val = np.quantile(vals_i, 0.75)
            max_val = vals_i.max()
            mean_val = vals_i.mean()
            table_data_abs.append([f"t{i}", f"{min_val:.4f}", f"{q1_val:.4f}", f"{median_val:.4f}", f"{q3_val:.4f}", f"{max_val:.4f}", f"{mean_val:.4f}"])
        else:
            table_data_abs.append([f"t{i}", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

    fig, ax = plt.subplots(figsize=(10, max(4, n_frames * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data_abs, colLabels=["Frame", "Min", "Q1", "Median", "Q3", "Max", "Mean"], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title(f"Per-frame absolute differences {arun} vs {brun}", pad=20)
    plt.savefig(perframe_table_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved", perframe_table_path)

    # New: per-frame percent differences (normalized by A scale)
    perframe_percent_vals = []
    for i in range(n_frames):
        absdiff_i = np.abs(A_raw[..., i] - B_raw[..., i])
        denom_i = A_raw[..., i]
        mask_i = M1 & M2 & (denom_i > 0.01) & (B_raw[..., i] > 0.01)
        percent_i = (absdiff_i[mask_i] / denom_i[mask_i]) * 100.0
        perframe_percent_vals.append(percent_i)

    plt.figure(figsize=(max(6, n_frames * 0.6), 4))
    plt.boxplot(perframe_percent_vals, showfliers=False)
    plt.xticks(list(range(1, n_frames + 1)), [f"t{i}" for i in range(n_frames)], rotation=45)
    plt.ylabel("Absolute difference (% of A)")
    plt.title(f"PET percent differences per frame {arun} vs {brun}")
    plt.tight_layout()
    plt.savefig(percent_boxplot_path, dpi=150)
    plt.close()
    print("Saved", percent_boxplot_path)

    # Table for per-frame percent differences
    table_data_pct = []
    for i, vals_i in enumerate(perframe_percent_vals):
        if len(vals_i) > 0:
            min_val = vals_i.min()
            q1_val = np.quantile(vals_i, 0.25)
            median_val = np.quantile(vals_i, 0.5)
            q3_val = np.quantile(vals_i, 0.75)
            max_val = vals_i.max()
            mean_val = vals_i.mean()
            table_data_pct.append([f"t{i}", f"{min_val:.2f}%", f"{q1_val:.2f}%", f"{median_val:.2f}%", f"{q3_val:.2f}%", f"{max_val:.2f}%", f"{mean_val:.2f}%"])
        else:
            table_data_pct.append([f"t{i}", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

    fig, ax = plt.subplots(figsize=(10, max(4, n_frames * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data_pct, colLabels=["Frame", "Min", "Q1", "Median", "Q3", "Max", "Mean"], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title(f"Per-frame percent differences {arun} vs {brun}", pad=20)
    plt.savefig(perframe_percent_table_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved", perframe_percent_table_path)
else:
    print("Data is not 4D; skipping per-frame boxplot.")
