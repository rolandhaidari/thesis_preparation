import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Comparing PET prep outputs: reference (run01) vs perturbed runs")

ref_run = "run01"
perturbed_runs = ["run04", "run09", "run15", "run16", "run17", "run19", "run20"]

# Load reference run
a_path = f"/home/roland/uni/thesisprep/ds001420-download/derivatives/petprep/{ref_run}/sub-dasb01/ses-baseline/pet/sub-dasb01_ses-baseline_space-T1w_desc-preproc_pet.nii.gz"
mask_ref_path = f"/home/roland/uni/thesisprep/ds001420-download/derivatives/petprep/{ref_run}/sub-dasb01/ses-baseline/pet/sub-dasb01_ses-baseline_space-T1w_desc-brain_mask.nii.gz"

a_ref = nib.load(a_path)
m_ref = nib.load(mask_ref_path)
A_raw_ref = a_ref.get_fdata(dtype=np.float32)
M_ref = m_ref.get_fdata(dtype=np.float32) > 0

# Determine if 4D
is_4d = A_raw_ref.ndim == 4
if is_4d:
    n_frames = A_raw_ref.shape[-1]
    print(f"Data is 4D with {n_frames} frames")
else:
    print("Data is 3D")

# Storage for per-frame statistics across all comparisons
if is_4d:
    all_perframe_abs = [[] for _ in range(n_frames)]  # per frame, list of arrays from all runs
    all_perframe_pct = [[] for _ in range(n_frames)]  # per frame, list of arrays from all runs

# Process each perturbed run
for brun in perturbed_runs:
    print(f"\n--- Processing {ref_run} vs {brun} ---")
    
    b_path = f"/home/roland/uni/thesisprep/ds001420-download/derivatives/petprep/{brun}/sub-dasb01/ses-baseline/pet/sub-dasb01_ses-baseline_space-T1w_desc-preproc_pet.nii.gz"
    mask_b_path = f"/home/roland/uni/thesisprep/ds001420-download/derivatives/petprep/{brun}/sub-dasb01/ses-baseline/pet/sub-dasb01_ses-baseline_space-T1w_desc-brain_mask.nii.gz"
    
    b = nib.load(b_path)
    m_b = nib.load(mask_b_path)
    B_raw = b.get_fdata(dtype=np.float32)
    M_b = m_b.get_fdata(dtype=np.float32) > 0
    
    # For averaged comparison
    A = A_raw_ref.copy()
    B = B_raw.copy()
    
    if A.ndim == 4:
        A = A.mean(axis=-1)
    if B.ndim == 4:
        B = B.mean(axis=-1)
    
    print("same shape:", A.shape == B.shape)
    print("same affine:", np.allclose(a_ref.affine, b.affine))
    
    diff = A - B
    absdiff = np.abs(diff)
    
    # intersection brain mask and exclude near zero background
    M = M_ref & M_b & (A > 0.01) & (B > 0.01)
    
    vals = absdiff[M].ravel()
    
    print(f"mean abs diff in mask: {vals.mean():.6f}")
    print(f"max abs diff in mask: {vals.max():.6f}")
    print(f"p95 abs diff in mask: {np.quantile(vals, 0.95):.6f}")
    
    # Per-frame collection only (no individual plots)
    if is_4d and A_raw_ref.ndim == 4 and B_raw.ndim == 4:
        for i in range(n_frames):
            diff_i = A_raw_ref[..., i] - B_raw[..., i]
            absdiff_i = np.abs(diff_i)
            mask_i = M_ref & M_b & (A_raw_ref[..., i] > 0.01) & (B_raw[..., i] > 0.01)
            vals_i = absdiff_i[mask_i].ravel()
            all_perframe_abs[i].append(vals_i)
            
            # Per-frame percent differences
            denom_i = A_raw_ref[..., i]
            percent_i = (absdiff_i[mask_i] / denom_i[mask_i]) * 100.0
            all_perframe_pct[i].append(percent_i)

# Aggregate statistics across all comparisons
if is_4d:
    print("\n=== AGGREGATE STATISTICS ACROSS ALL PERTURBATIONS ===\n")
    
    # Per-frame mean and std for absolute differences
    agg_mean_abs = []
    agg_std_abs = []
    table_data_agg_abs = []
    
    for i in range(n_frames):
        if len(all_perframe_abs[i]) > 0:
            # Concatenate all voxel values from all perturbations for frame i
            all_vals_i = np.concatenate(all_perframe_abs[i])
            mean_val = all_vals_i.mean()
            std_val = all_vals_i.std()
            agg_mean_abs.append(mean_val)
            agg_std_abs.append(std_val)
            
            # Compute quantiles
            q1 = np.quantile(all_vals_i, 0.25)
            q2 = np.quantile(all_vals_i, 0.5)
            q3 = np.quantile(all_vals_i, 0.75)
            
            table_data_agg_abs.append([f"t{i}", f"{mean_val:.4f}", f"{std_val:.4f}", f"{q1:.4f}", f"{q2:.4f}", f"{q3:.4f}"])
    
    # Per-frame mean and std for percent differences
    agg_mean_pct = []
    agg_std_pct = []
    table_data_agg_pct = []
    
    for i in range(n_frames):
        if len(all_perframe_pct[i]) > 0:
            all_vals_i = np.concatenate(all_perframe_pct[i])
            mean_val = all_vals_i.mean()
            std_val = all_vals_i.std()
            agg_mean_pct.append(mean_val)
            agg_std_pct.append(std_val)
            
            q1 = np.quantile(all_vals_i, 0.25)
            q2 = np.quantile(all_vals_i, 0.5)
            q3 = np.quantile(all_vals_i, 0.75)
            
            table_data_agg_pct.append([f"t{i}", f"{mean_val:.2f}%", f"{std_val:.2f}%", f"{q1:.2f}%", f"{q2:.2f}%", f"{q3:.2f}%"])
    
    # Plot mean ± std for absolute differences
    fig, ax = plt.subplots(figsize=(12, 5))
    frames = np.arange(n_frames)
    agg_mean_abs = np.array(agg_mean_abs)
    agg_std_abs = np.array(agg_std_abs)
    
    ax.plot(frames, agg_mean_abs, 'o-', linewidth=2, markersize=8, label='Mean')
    ax.fill_between(frames, agg_mean_abs - agg_std_abs, agg_mean_abs + agg_std_abs, alpha=0.3, label='±1 Std Dev')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Absolute difference')
    ax.set_title(f'Mean ± Std per frame across all perturbations vs {ref_run}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./plots/aggregate_mean_std_abs_petref_{ref_run}.png", dpi=150)
    plt.close()
    print(f"Saved ./plots/aggregate_mean_std_abs_petref_{ref_run}.png")
    
    # Plot mean ± std for percent differences
    fig, ax = plt.subplots(figsize=(12, 5))
    agg_mean_pct = np.array(agg_mean_pct)
    agg_std_pct = np.array(agg_std_pct)
    
    ax.plot(frames, agg_mean_pct, 'o-', linewidth=2, markersize=8, label='Mean', color='orange')
    ax.fill_between(frames, agg_mean_pct - agg_std_pct, agg_mean_pct + agg_std_pct, alpha=0.3, label='±1 Std Dev', color='orange')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Percent difference (%)')
    ax.set_title(f'Mean ± Std percent difference per frame across all perturbations vs {ref_run}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./plots/aggregate_mean_std_pct_petref_{ref_run}.png", dpi=150)
    plt.close()
    print(f"Saved ./plots/aggregate_mean_std_pct_petref_{ref_run}.png")
    
    # Save aggregate statistics tables
    fig, ax = plt.subplots(figsize=(10, max(4, n_frames * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data_agg_abs, colLabels=["Frame", "Mean", "Std", "Q1", "Median", "Q3"], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title(f'Aggregate absolute differences per frame across all perturbations', pad=20)
    plt.savefig(f"./plots/aggregate_table_abs_petref_{ref_run}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ./plots/aggregate_table_abs_petref_{ref_run}.png")
    
    fig, ax = plt.subplots(figsize=(10, max(4, n_frames * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data_agg_pct, colLabels=["Frame", "Mean", "Std", "Q1", "Median", "Q3"], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title(f'Aggregate percent differences per frame across all perturbations', pad=20)
    plt.savefig(f"./plots/aggregate_table_pct_petref_{ref_run}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ./plots/aggregate_table_pct_petref_{ref_run}.png")
