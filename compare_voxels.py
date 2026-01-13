import nibabel as nib
import numpy as np
a_path = "/data/derivatives/petprep/run02/sub-dasb01/ses-baseline/pet/sub-dasb01_ses-baseline_space-MNI152NLin2009cAsym_desc-preproc_pet.nii.gz"
b_path = "/data/derivatives/petprep/run03_fuzzy/sub-dasb01/ses-baseline/pet/sub-dasb01_ses-baseline_space-MNI152NLin2009cAsym_desc-preproc_pet.nii.gz"
mask_path = "/data/derivatives/petprep/run02/sub-dasb01/ses-baseline/pet/sub-dasb01_ses-baseline_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
a = nib.load(a_path)
b = nib.load(b_path)
m = nib.load(mask_path)
A = a.get_fdata(dtype=np.float32)
B = b.get_fdata(dtype=np.float32)
M = m.get_fdata(dtype=np.float32) > 0
diff = (A - B)
absdiff = np.abs(diff)
print("mean abs diff in mask:", absdiff[M].mean())
print("max abs diff in mask:", absdiff[M].max())
print("p95 abs diff in mask:", np.quantile(absdiff[M], 0.95))
out = nib.Nifti1Image(absdiff.astype(np.float32), a.affine, a.header)
nib.save(out, "/scripts/absdiff_MNI_preproc_pet_1_3_fuzzy.nii.gz")
print("Saved abs diff image to /scripts/absdiff_MNI_preproc_pet_1_3_fuzzy.nii.gz")