import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.processing import resample_to_output, resample_from_to
from pathlib import Path

print("Comparing local PET prep outputs")
arun="run04"
brun="run18_fuzzy"

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
if A_raw.shape[:3] != B_raw.shape[:3] or not np.allclose(a.affine, b.affine):
    print("Resampling B to A grid ...")
    
    # Ensure mask is 3D before resampling
    
    m2_data = m2.get_fdata(dtype=np.float32)

    if m2_data.ndim == 4:
        print("Mask is 4D, collapsing to 3D")
        m2_data = np.any(m2_data > 0, axis=-1).astype(np.float32)

    m2_3d = nib.Nifti1Image(m2_data, m2.affine)
    
    # Create a 3D reference image from the 4D PET image
    # Extract the 3D spatial affine from the 4D/5D image
    a_affine_3d = a.affine[:3, :3].copy()
    a_affine_3d = np.eye(4)
    a_affine_3d[:3, :3] = a.affine[:3, :3]
    a_affine_3d[:3, 3] = a.affine[:3, 3]
    
    # Create a 3D reference image with same dimensions as first 3 dimensions of A
    a_ref_3d = nib.Nifti1Image(np.zeros(A_raw.shape[:3]), a_affine_3d)
    
    # Now resample mask to the 3D reference
    m2_resampled = resample_from_to(m2_3d, a_ref_3d)
    M2 = m2_resampled.get_fdata(dtype=np.float32) > 0

else:
    M2 = m2.get_fdata(dtype=np.float32) > 0



print(f"A shape: {A.shape}, B shape: {B.shape}")
print(f"M1 shape: {M1.shape}, M2 shape: {M2.shape}")

# Important: After resampling, ensure A and B have the same shape for mask intersection
if A.shape != B.shape:
    print(f"Warning: A shape {A.shape} != B shape {B.shape}. Trimming to match.")
    min_shape = tuple(min(sa, sb) for sa, sb in zip(A.shape, B.shape))
    A = A[:min_shape[0], :min_shape[1], :min_shape[2]]
    B = B[:min_shape[0], :min_shape[1], :min_shape[2]]
    M1 = M1[:min_shape[0], :min_shape[1], :min_shape[2]]
    M2 = M2[:min_shape[0], :min_shape[1], :min_shape[2]]


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

# New: per-frame boxplot without averaging (timesteps 15-25 only)
if A_raw.ndim == 4 and B_raw.ndim == 4:
    n_frames = A_raw.shape[-1]
    frame_start = 14  # timestep 15 (0-indexed)
    frame_end = 25    # timestep 25 (0-indexed), inclusive
    frame_range = range(frame_start, min(frame_end + 1, n_frames))
    perframe_vals = []
    for i in frame_range:
        A_frame = A_raw[..., i]
        B_frame = B_raw[..., i]
        
        # Ensure frames have same shape
        if A_frame.shape != B_frame.shape:
            min_shape = tuple(min(sa, sb) for sa, sb in zip(A_frame.shape, B_frame.shape))
            A_frame = A_frame[:min_shape[0], :min_shape[1], :min_shape[2]]
            B_frame = B_frame[:min_shape[0], :min_shape[1], :min_shape[2]]
        diff_i = A_frame - B_frame
        absdiff_i = np.abs(diff_i)
        mask_i = M1 & M2 & (A_frame > 0.01) & (B_frame > 0.01)
        vals_i = absdiff_i[mask_i].ravel()
        perframe_vals.append(vals_i)

    plt.figure(figsize=(max(6, len(perframe_vals) * 0.6), 4))
    plt.boxplot(perframe_vals, showfliers=False)
    plt.xticks(list(range(1, len(perframe_vals) + 1)), [f"t{i}" for i in frame_range], rotation=45)
    plt.ylabel("Absolute difference")
    plt.title(f"PET absolute differences per frame {arun} vs {brun}")
    plt.tight_layout()
    plt.savefig(perframe_boxplot_path, dpi=150)
    plt.close()
    print("Saved", perframe_boxplot_path)

    # Table for per-frame absolute differences
    table_data_abs = []
    for idx, i in enumerate(frame_range):
        vals_i = perframe_vals[idx]
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

    # New: per-frame percent differences (normalized by A scale, timesteps 15-25 only)
    perframe_percent_vals = []
    for i in frame_range:
        A_frame = A_raw[..., i]
        B_frame = B_raw[..., i]
        
        # Ensure frames have same shape
        if A_frame.shape != B_frame.shape:
            min_shape = tuple(min(sa, sb) for sa, sb in zip(A_frame.shape, B_frame.shape))
            A_frame = A_frame[:min_shape[0], :min_shape[1], :min_shape[2]]
            B_frame = B_frame[:min_shape[0], :min_shape[1], :min_shape[2]]

        absdiff_i = np.abs(A_frame - B_frame)
        denom_i = A_frame
        mask_i = M1 & M2 & (denom_i > 0.01) & (B_frame > 0.01)
        percent_i = (absdiff_i[mask_i] / denom_i[mask_i]) * 100.0
        perframe_percent_vals.append(percent_i)

    plt.figure(figsize=(max(6, len(perframe_percent_vals) * 0.6), 4))
    plt.boxplot(perframe_percent_vals, showfliers=False)
    plt.xticks(list(range(1, len(perframe_percent_vals) + 1)), [f"t{i}" for i in frame_range], rotation=45)
    plt.ylabel("Absolute difference (% of A)")
    plt.title(f"PET percent differences per frame {arun} vs {brun}")
    plt.tight_layout()
    plt.savefig(percent_boxplot_path, dpi=150)
    plt.close()
    print("Saved", percent_boxplot_path)

    # Table for per-frame percent differences
    table_data_pct = []
    for idx, i in enumerate(frame_range):
        vals_i = perframe_percent_vals[idx]
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
