import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

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

A = a.get_fdata(dtype=np.float32)
B = b.get_fdata(dtype=np.float32)

if A.ndim == 4:
    print("A is 4D, averaging across frames")
    A = A.mean(axis=-1)

if B.ndim == 4:
    print("B is 4D, averaging across frames")
    B = B.mean(axis=-1)


M1 = m1.get_fdata(dtype=np.float32) > 0
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

absdiff_path = f"./absdiff_petref_{tag}.nii.gz"
clip_path = f"./absdiff_petref_{tag}_clip99.nii.gz"
boxplot_path = f"./absdiff_boxplot_petref_{tag}.png"

out = nib.Nifti1Image(absdiff.astype(np.float32), a.affine, a.header)
nib.save(out, absdiff_path)

clip_thr = np.quantile(vals, 0.99)
clipped = np.clip(absdiff, 0, clip_thr)
nib.save(nib.Nifti1Image(clipped.astype(np.float32), a.affine, a.header), clip_path)

plt.figure()
plt.boxplot([vals], showfliers=False)
plt.xticks([1], ["abs diff in common brain mask"])
plt.ylabel("Absolute difference")
plt.title(f"PET absolute differences {arun} vs {brun}")
plt.savefig(boxplot_path, dpi=150)
plt.close()

print("Saved", absdiff_path)
print("Saved", clip_path)
print("Saved", boxplot_path)
