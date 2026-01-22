import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to
from pathlib import Path

print("Per-frame summed PET signal: run26 vs run27")
arun = "run26"
brun = "run27"
subject = "sub-dasb01"
session = "ses-baseline"

base = "/home/roland/uni/thesisprep/ds001420-new/derivatives/petprep"
a_path = f"{base}/{arun}/{subject}/{session}/pet/{subject}_{session}_space-T1w_desc-preproc_pet.nii.gz"
b_path = f"{base}/{brun}/{subject}/{session}/pet/{subject}_{session}_space-T1w_desc-preproc_pet.nii.gz"
mask1_path = f"{base}/{arun}/{subject}/{session}/pet/{subject}_{session}_space-T1w_desc-brain_mask.nii.gz"
mask2_path = f"{base}/{brun}/{subject}/{session}/pet/{subject}_{session}_space-T1w_desc-brain_mask.nii.gz"

# Load data
A_img = nib.load(a_path)
B_img = nib.load(b_path)
M1_img = nib.load(mask1_path)
M2_img = nib.load(mask2_path)

A_raw = A_img.get_fdata(dtype=np.float32)
B_raw = B_img.get_fdata(dtype=np.float32)

# Prepare masks as 3D boolean arrays
m1_data = M1_img.get_fdata(dtype=np.float32)
if m1_data.ndim == 4:
    m1_data = np.any(m1_data > 0, axis=-1).astype(np.float32)
M1 = m1_data > 0

m2_data = M2_img.get_fdata(dtype=np.float32)
if m2_data.ndim == 4:
    m2_data = np.any(m2_data > 0, axis=-1).astype(np.float32)
M2_3d = nib.Nifti1Image(m2_data, M2_img.affine)
M1_3d = nib.Nifti1Image(m1_data, M1_img.affine)
M2_resampled = resample_from_to(M2_3d, M1_3d)
M2 = M2_resampled.get_fdata(dtype=np.float32) > 0

# Intersection mask
M = M1 & M2

print(f"A_raw shape: {A_raw.shape}, B_raw shape: {B_raw.shape}")

# Determine frames
nA = A_raw.shape[-1] if A_raw.ndim == 4 else 1
nB = B_raw.shape[-1] if B_raw.ndim == 4 else 1
n_frames = min(nA, nB)

# Threshold to avoid background contributions
thr = 0.01

sums_A = []
sums_B = []

for i in range(n_frames):
    A_frame = A_raw[..., i] if A_raw.ndim == 4 else A_raw
    B_frame = B_raw[..., i] if B_raw.ndim == 4 else B_raw

    # Align shapes by cropping to common min shape
    min_shape = tuple(min(sa, sb, sm) for sa, sb, sm in zip(A_frame.shape, B_frame.shape, M.shape))
    A_f = A_frame[:min_shape[0], :min_shape[1], :min_shape[2]]
    B_f = B_frame[:min_shape[0], :min_shape[1], :min_shape[2]]
    M_f = M[:min_shape[0], :min_shape[1], :min_shape[2]]

    mask_i = M_f & (A_f > thr) & (B_f > thr)
    sums_A.append(float(A_f[mask_i].sum()))
    sums_B.append(float(B_f[mask_i].sum()))

# Prepare output dirs
plots_hist_dir = Path("./plots/histogram")
plots_hist_dir.mkdir(parents=True, exist_ok=True)

# Plot grouped bar chart
frames = np.arange(n_frames)
width = 0.45
plt.figure(figsize=(max(8, n_frames * 0.5), 5))
plt.bar(frames - width/2, sums_A, width, label=arun, color="tab:blue")
plt.bar(frames + width/2, sums_B, width, label=brun, color="tab:orange")
plt.xticks(frames, [f"t{i}" for i in range(n_frames)], rotation=45)
plt.ylabel("Summed PET signal (masked)")
plt.title(f"Per-frame summed PET signal {arun} vs {brun}")
plt.legend()
plt.tight_layout()
outfile = plots_hist_dir / f"summed_{arun}_vs_{brun}.png"
plt.savefig(outfile, dpi=150)
plt.close()

print("Saved", outfile)
print("Frames:", n_frames)
print("First 5 sums run26:", sums_A[:5])
print("First 5 sums run27:", sums_B[:5])

# Relative error histogram (per-frame, based on summed signals)
rel_errors = []
for i in range(n_frames):
    denom = sums_A[i]
    if denom > 1e-8:
        rel_errors.append(((sums_B[i] - denom) / denom) * 100.0)

plots_hist_dir.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8, 5))
plt.hist(rel_errors, bins=20, color="tab:purple", edgecolor="white")
plt.xlabel("Relative error (%) (run27 vs run26)")
plt.ylabel("Count")
plt.title(f"Per-frame relative error of summed signal {arun} vs {brun}")
if len(rel_errors) > 0:
    mean_rel = float(np.mean(rel_errors))
    median_rel = float(np.median(rel_errors))
    plt.axvline(mean_rel, color="k", linestyle="--", label=f"Mean {mean_rel:.2f}%")
    plt.axvline(median_rel, color="gray", linestyle=":", label=f"Median {median_rel:.2f}%")
    plt.legend()
plt.tight_layout()
outfile_rel = plots_hist_dir / f"relerr_{arun}_vs_{brun}.png"
plt.savefig(outfile_rel, dpi=150)
plt.close()
print("Saved", outfile_rel)
