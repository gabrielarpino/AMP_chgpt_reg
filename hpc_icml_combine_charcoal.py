import os
import argparse
from matplotlib import pyplot as plt
# import tikzplotlib
import numpy as np

# different jobs for different sigma
# jobs for L=2, p=300, sigma=0.2,0.3,0.5, fixed SNR
jobs = ["42415461", "42415497", "42415557"] 
# jobs for L=3, p=300, sigma=0.2,0.3,0.5, fixed SNR
# jobs = ["42415624", "42415696", "42415753"]
σ_arr = [0.2, 0.3, 0.5]
num_σs = len(jobs)
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
                    Lmax = data['Lmax']
                    σ_w = data['σ_w'] # difference between adjacent signals

                    T = data['T']
                    num_trials = data['num_trials']
                    # chgpt_loc_str_list = ['1/3', '8/15'] # data['chgpt_loc_str_list'] 1/3+1/5
                    break

num_delta = 10
δ_arr = np.zeros((num_σs, num_delta))
amp_hausdorff_all = np.zeros((num_σs, num_delta, num_trials)) 
# Make this 1 when no chgpts were detected
amp_hausdorff_all[:] = np.nan
se_hausdorff_all = np.zeros((num_σs, num_delta, num_trials))
se_hausdorff_all[:] = np.nan
charcoal_hausdorff_all = np.zeros((num_σs, num_delta, num_trials))
charcoal_hausdorff_all[:] = np.nan

amp_runtime_all = np.zeros((num_σs, num_delta, num_trials))
amp_runtime_all[:] = np.nan
charcoal_runtime_all = np.zeros((num_σs, num_delta, num_trials))
charcoal_runtime_all[:] = np.nan

for i_job, (job_num, σ) in enumerate(zip(jobs, σ_arr)):
    print(f"[{i_job}/{num_σs}] Processing job {job_num}, σ = {σ}")
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
                        se_hausdorff_all[i_job, delta_idx, :] = data['se_hausdorff']
                        charcoal_hausdorff_all[i_job, delta_idx, :] = data['charcoal_hausdorff']

                        amp_runtime_all[i_job, delta_idx] = data['amp_runtime']
                        charcoal_runtime_all[i_job, delta_idx] = data['charcoal_runtime']
                        i += 1
            assert i == num_delta
            

# Replace infinite hausdorff with n/n=1 by adding [1,n] to the set of chgpts:
amp_hausdorff_all[np.isinf(amp_hausdorff_all)] = 1
se_hausdorff_all[np.isinf(se_hausdorff_all)] = 1
charcoal_hausdorff_all[np.isinf(charcoal_hausdorff_all)] = 1

assert np.allclose(np.mean(δ_arr, axis=0), δ_arr[0])
icml_path = './icml/'
# cmap = plt.get_cmap('RdBu') 
# colors = [cmap(i) for i in [0, 0.2, 0.8]]
# colors = ['C0', 'C1', 'C2']
colors = ['#66c2a5','#fc8d62','#8da0cb'] # light
# colors = ['#1b9e77','#d95f02','#7570b3'] # dark

print("====== Plotting Hausdorff (ignoring nans) ======")
fig = plt.figure(figsize = (8,5))
plt.rcParams.update({'font.size': 16})
plt.subplot(2,1,1)
for i_σ, σ in enumerate(σ_arr):
    plt.errorbar(δ_arr[0], np.mean(amp_hausdorff_all[i_σ], axis=-1),
                    yerr=np.std(amp_hausdorff_all[i_σ], axis=-1),
                    label=f'AMP, σ={σ}', color=colors[i_σ], alpha=0.8, 
                    linestyle='--', linewidth=2)
    # plt.scatter(δ_arr[0], np.mean(se_hausdorff_all[i_σ], axis=-1),
    #                 label=f'Theory, σ={σ}', color=f'C{i_σ}', marker='x')
    plt.errorbar(δ_arr[0], np.mean(charcoal_hausdorff_all[i_σ], axis=-1),
                    yerr=np.std(charcoal_hausdorff_all[i_σ], axis=-1),
                    label=f'Charcoal, σ={σ}', color=colors[i_σ], linewidth=2)
plt.ylim([0, 1.2])
# plt.xlabel('δ = n/p')
plt.ylabel('Hausdorff distance')
# plt.yscale('log')
ax = plt.gca()
ax.set_xticks([])
# plt.legend()
# plt.savefig(os.path.join(icml_path, f'icml_charcoal_hausdorff_v_delta_L{L}_L{Lmax}_p{p}_sigma_{σ}_alpha_{α}_σw_{σ_w}.pdf'))

print("====== Plotting Runtime (ignoring nans) ======")
# plt.figure(figsize = (8,3))
plt.subplot(2,1,2)
for i_σ, σ in enumerate(σ_arr):
    plt.errorbar(δ_arr[0], np.mean(amp_runtime_all[i_σ], axis=-1),
                    yerr=np.std(amp_runtime_all[i_σ], axis=-1),
                    label=f'AMP, σ={σ}', color=colors[i_σ], alpha=0.8, 
                    linestyle='--', linewidth=2)
    plt.errorbar(δ_arr[0], np.mean(charcoal_runtime_all[i_σ], axis=-1),
                    yerr=np.std(charcoal_runtime_all[i_σ], axis=-1),    
                    label=f'Charcoal, σ={σ}', color=colors[i_σ], linewidth=2)
plt.xlabel('δ = n/p')
plt.ylabel('Runtime (s)')
plt.legend(fontsize=12)
# plt.savefig(os.path.join(icml_path, f'icml_charcoal_runtime_v_delta_L{L}_L{Lmax}_p{p}_sigma_{σ}_alpha_{α}_σw_{np.round(σ_w, 2)}.pdf'))
plt.tight_layout()
fig.subplots_adjust(hspace=0.05, wspace=0.025)
plt.savefig(os.path.join(icml_path, f'icml_charcoal_L{L}_L{Lmax}_p{p}_alpha_{α}_σw_{np.round(σ_w, 2)}.pdf'))
print("====== Done ======") 

