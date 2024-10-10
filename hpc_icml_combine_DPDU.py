import os
import argparse
from matplotlib import pyplot as plt
# import tikzplotlib
import numpy as np

# jobs for L=2, sigma=0.1, p=200, alpha=0.5, fixed SNR
jobs = ["42416418", "42416417", "42416441"]
# jobs for L=3, sigma=0.1, p=200, alpha=0.5, fixed SNR
# jobs = ["42416462", "42416463", "42416464"]
num_jobs = len(jobs)
save_path = './hpc_results/'

for dir_name in os.listdir(save_path):
    dir_path = os.path.join(save_path, dir_name)
    if os.path.isdir(dir_path) and \
        dir_name.startswith(jobs[0]):
        # Initialise arrays:
        for filename in os.listdir(dir_path):
            if filename.endswith('.npz'):
                with np.load(os.path.join(dir_path, filename)) as data:
                    # Values below don't change with δ
                    p = data['p']
                    α = data['α']
                    L = data['L']
                    σ = data['σ'] # additive noise
                    T = data['T']
                    num_trials = data['num_trials']
                    # chgpt_loc_str_list = ['1/3', '8/15'] # data['chgpt_loc_str_list'] 1/3+1/5
                    break

num_delta = 10
δ_arr = np.zeros((num_jobs, num_delta))
amp_hausdorff_all = np.zeros((num_jobs, num_delta, num_trials)) 
# Make this 1 when no chgpts were detected
amp_hausdorff_all[:] = np.nan
se_hausdorff_all = np.zeros((num_jobs, num_delta, num_trials))
se_hausdorff_all[:] = np.nan
DPDU_hausdorff_all = np.zeros((num_jobs, num_delta, num_trials))
DPDU_hausdorff_all[:] = np.nan
DCDP_hausdorff_all = np.zeros((num_jobs, num_delta, num_trials))
DCDP_hausdorff_all[:] = np.nan
DP_hausdorff_all = np.zeros((num_jobs, num_delta, num_trials))
DP_hausdorff_all[:] = np.nan

amp_runtime_all = np.zeros((num_jobs, num_delta, num_trials))
amp_runtime_all[:] = np.nan
DPDU_runtime_all = np.zeros((num_jobs, num_delta, num_trials))
DPDU_runtime_all[:] = np.nan
DCDP_runtime_all = np.zeros((num_jobs, num_delta, num_trials))
DCDP_runtime_all[:] = np.nan
DP_runtime_all = np.zeros((num_jobs, num_delta, num_trials))
DP_runtime_all[:] = np.nan

for i_job, job_num in enumerate(jobs):
    print(f"Processing job {job_num}")
    for dir_name in os.listdir(save_path):
        dir_path = os.path.join(save_path, dir_name)
        if os.path.isdir(dir_path) and \
            dir_name.startswith(jobs[i_job]):
            print(f"Found data for job {job_num}")
            i = 0
            for filename in os.listdir(dir_path):
                if filename.endswith('.npz'):
                    with np.load(os.path.join(dir_path, filename)) as data:
                        delta_idx = data['delta_idx']
                        δ_arr[i_job, delta_idx] = data['δ']
                        amp_hausdorff_all[i_job, delta_idx,:] = data['amp_hausdorff']
                        se_hausdorff_all[i_job, delta_idx, :] = data['se_fixed_C_hausdorff']
                        DPDU_hausdorff_all[i_job, delta_idx] = data['DPDU_hausdorff']
                        DCDP_hausdorff_all[i_job, delta_idx] = data['DCDP_hausdorff']
                        DP_hausdorff_all[i_job, delta_idx] = data['DP_hausdorff']

                        amp_runtime_all[i_job, delta_idx] = data['amp_runtime']
                        DPDU_runtime_all[i_job, delta_idx] = data['DPDU_runtime']
                        DCDP_runtime_all[i_job, delta_idx] = data['DCDP_runtime']
                        DP_runtime_all[i_job, delta_idx] = data['DP_runtime']
                        i += 1
            assert i == num_delta
            

# Replace infinite hausdorff with n/n=1 by adding [1,n] to the set of chgpts:
amp_hausdorff_all[np.isinf(amp_hausdorff_all)] = 1
se_hausdorff_all[np.isinf(se_hausdorff_all)] = 1
DPDU_hausdorff_all[np.isinf(DPDU_hausdorff_all)] = 1
DCDP_hausdorff_all[np.isinf(DCDP_hausdorff_all)] = 1
DP_hausdorff_all[np.isinf(DP_hausdorff_all)] = 1

assert np.allclose(np.mean(δ_arr, axis=0), δ_arr[0])
icml_path = './icml/'
plt.rcParams.update({'font.size': 16})
# cmap = plt.get_cmap('RdBu') 
# colors = [cmap(i) for i in [0, 0.2, 0.7, 0.9]]
# colors = ['C0', 'C1', 'C2', 'C3']
colors = ['black', '#66c2a5','#fc8d62','#8da0cb'] # light
# colors = ['black', '#1b9e77','#d95f02','#7570b3'] # dark

print("====== Plotting Hausdorff (ignoring nans) ======")
fig = plt.figure(figsize = (8,5))
plt.subplot(2,1,1)
plt.errorbar(δ_arr[0], np.mean(amp_hausdorff_all, axis=(0, 2)),
                yerr=np.std(amp_hausdorff_all, axis=(0, 2)),
                label='AMP', color=colors[0], linestyle='--', linewidth=2)
# plt.errorbar(δ_arr[0], np.mean(se_hausdorff_all, axis=(0, 2)),
#                 yerr=np.std(se_hausdorff_all, axis=(0, 2)),
#                 label='SE', color='C0', linestyle='--')
plt.errorbar(δ_arr[0], np.nanmean(DPDU_hausdorff_all, axis=(0, 2)),
                yerr=np.nanstd(DPDU_hausdorff_all, axis=(0, 2)),
                label='DPDU', color=colors[1], linewidth=2)
plt.errorbar(δ_arr[0], np.nanmean(DCDP_hausdorff_all, axis=(0, 2)),
                yerr=np.nanstd(DCDP_hausdorff_all, axis=(0, 2)),
                label='DCDP', color=colors[2], linewidth=2)
plt.errorbar(δ_arr[0], np.nanmean(DP_hausdorff_all, axis=(0, 2)),
                yerr=np.nanstd(DP_hausdorff_all, axis=(0, 2)),
                label='DP', color=colors[3], linewidth=2)
# plt.ylim([0, 1])
# plt.xlabel('δ = n/p')
plt.ylabel('Hausdorff distance')
ax = plt.gca()
ax.set_xticks([])
# plt.legend()
# plt.savefig(os.path.join(icml_path, f'icml_DPDU_hausdorff_v_delta_L_{L}_p_{p}_sigma_{σ}_alpha_{α}.pdf'))

print("====== Plotting Runtime (ignoring nans) ======")
# plt.figure()
plt.subplot(2,1,2)
plt.errorbar(δ_arr[0], np.mean(amp_runtime_all, axis=(0, 2)),
                yerr=np.std(amp_runtime_all, axis=(0, 2)),
                label='AMP', color=colors[0], linestyle='--', linewidth=2)
plt.errorbar(δ_arr[0], np.nanmean(DPDU_runtime_all, axis=(0, 2)),
                yerr=np.nanstd(DPDU_runtime_all, axis=(0, 2)),
                label='DPDU', color=colors[1], linewidth=2)
plt.errorbar(δ_arr[0], np.nanmean(DCDP_runtime_all, axis=(0, 2)),
                yerr=np.nanstd(DCDP_runtime_all, axis=(0, 2)),
                label='DCDP', color=colors[2], linewidth=2)
plt.errorbar(δ_arr[0], np.nanmean(DP_runtime_all, axis=(0, 2)),
                yerr=np.nanstd(DP_runtime_all, axis=(0, 2)),
                label='DP', color=colors[3], linewidth=2)
plt.xlabel('δ = n/p')
plt.ylabel('Runtime (s)')
plt.legend()
# plt.savefig(os.path.join(icml_path, f'icml_DPDU_runtime_v_delta_L_{L}_p_{p}_sigma_{σ}_alpha_{α}.pdf'))
plt.tight_layout()
fig.subplots_adjust(hspace=0.05, wspace=0.025)
plt.savefig(os.path.join(icml_path, f'icml_DPDU_L_{L}_p_{p}_sigma_{σ}_alpha_{α}.pdf'))

################# Plotting Hausdorff for AMP with Lmax=2 and Lmax=3 vs DCDP #################
if L == 2:
    # Read extra AMP data for Lmax=3, L=2:
    for dir_name in os.listdir(save_path):
        dir_path = os.path.join(save_path, dir_name)
        if os.path.isfile(dir_path) and \
            dir_name.startswith('chgpts_sparse_Lmax3_L2_2024-01-31_17'):
            with np.load(dir_path) as data:
                # Values below don't change with δ
                p_Lmax3 = data['p']
                δ_arr = data['δ_arr']
                L_Lmax3 = data['L']
                Lmax = data['Lmax']
                Lmin = data['Lmin']
                σ_Lmax3 = data['σ'] # noise std
                α_Lmax3 = data['α'] # sparsity parameter, fraction of nonzeros
                σ_l_arr_unscaled = data['σ_l_arr_unscaled'] # signal std
                T = data['T']
                num_trials = data['num_trials']
                amp_hausdorff_Lmax3 = data['amp_hausdorff']
                se_fixed_C_hausdorff_Lmax3 = data['se_fixed_C_hausdorff']
                amp_runtime_Lmax3 = data['amp_runtime']
                η_true = data['η_true']
                frac_Delta = data['frac_Delta']
    assert p == p_Lmax3 and σ == σ_Lmax3 and α == α_Lmax3 and L == L_Lmax3
    fig = plt.figure(figsize = (8,5))
    plt.subplot(2,1,1)
    plt.errorbar(δ_arr, np.mean(amp_hausdorff_all, axis=(0, 2)),
                    yerr=np.std(amp_hausdorff_all, axis=(0, 2)),
                    label='AMP,' + r' $L^*$=2, $L$=2', color=colors[0], 
                    linestyle='--', linewidth=2)
    plt.errorbar(δ_arr, np.mean(amp_hausdorff_Lmax3, axis=1),
                    yerr=np.std(amp_hausdorff_Lmax3, axis=1),
                    label='AMP,' + r' $L^*$=2, $L$=3', color=colors[0], linewidth=2)
    # plot DCDP only:
    plt.errorbar(δ_arr, np.nanmean(DCDP_hausdorff_all, axis=(0, 2)),
                    yerr=np.nanstd(DCDP_hausdorff_all, axis=(0, 2)),
                    label='DCDP', color=colors[2], linewidth=2)
    plt.legend()
    plt.ylabel('Hausdorff distance')
    ax = plt.gca()
    ax.set_xticks([])

    plt.subplot(2,1,2)
    plt.errorbar(δ_arr, np.mean(amp_runtime_all, axis=(0, 2)),
                    yerr=np.std(amp_runtime_all, axis=(0, 2)),
                    label='AMP,' + r' $L^*$=2, $L$=2', color=colors[0], 
                    linestyle='--', linewidth=2)
    plt.errorbar(δ_arr, np.mean(amp_runtime_Lmax3, axis=1),
                    yerr=np.std(amp_runtime_Lmax3, axis=1),
                    label='AMP,' + r' $L^*$=2, $L$=3', color=colors[0], linewidth=2)
    plt.errorbar(δ_arr, np.nanmean(DCDP_runtime_all, axis=(0, 2)),
                    yerr=np.nanstd(DCDP_runtime_all, axis=(0, 2)),
                    label='DCDP', color=colors[2], linewidth=2)
    plt.xlabel('δ = n/p')
    plt.ylabel('Runtime (s)')
    # plt.legend()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.05, wspace=0.025)
    plt.savefig(os.path.join(icml_path, f'icml_DCDP_AMP_Lmax_{Lmax}_L_{L_Lmax3}_p_{p}_sigma_{σ}_alpha_{α}.pdf'))

print("====== Done ======") 

