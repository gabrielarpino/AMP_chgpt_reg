import rpy2

# os.environ["R_HOME"] = "/usr/lib64/R"
from rpy2.robjects.packages import importr

from amp.posterior import MAP_η, posterior_over_η, η_to_ψ
# utils = importr('utils')
# base = importr('base')
# chgpts = importr('changepoints')
# charcoal = importr('charcoal')
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from amp.comparison import run_charcoal

import numpy as np
import jax.numpy as jnp
import numpy.random as nprandom
from matplotlib import pyplot as plt
from amp.signal_priors import SparseDiffSignal
from amp.marginal_separable_jax import GAMP, SE_fixed_C_v1, q, GAMP_full
from amp import hausdorff
from amp.signal_configuration import unif_prior_to_η_φ
import argparse
import datetime
from tqdm.auto import tqdm
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        'Run charcoal and AMP on sparse diff signals for" + \
            " varying oversampling ratio delta.')
    parser.add_argument('--num_delta', type=int, default=10,
        help="Number of delta values or number of jobs in job array.")
    parser.add_argument('--delta_idx', type=int, default=0,
        help="Index of delta in the array of delta values to use.")
    parser.add_argument('--p', type=int, default=300, help="Number of covariates.")
    parser.add_argument('--sigma_w', type=float, default=20,
        help="Value of sigma_w, which determines the corr(beta1, beta2)" + \
             " and the l2-norm of the sparse change vector. Higher sigma_w" + \
                " means smaller correlation and larger jump size so" + \
                    " strictly easier detection.")
    parser.add_argument('--sigma', type=float, default=0.2, help="noise std.")
    parser.add_argument('--L', type=int, default=3, help="true number of signals.")
    parser.add_argument('--Lmax', type=int, default=3, 
                        help="Maximum number of signals to be considered in prior ϕ. " + \
                            "Max number of chgpts = Lmax-1. " + \
                            "By default make ground truth L=Lmax.")
    parser.add_argument('--frac_Delta', type=float, default=0.03,
                        help="fraction of n to use as min distance between chgpts.")
    parser.add_argument('--uniform_p_l', action='store_true',
        help="Use uniform prior p_l over number of signals. store_true by default " + 
                            "i.e. use uniform prior over configs by default "+
                            "in which case no need to pass in anything." + 
                            "To employ uniform prior over number of signals, " +
                            "pass in --uniform_p_l.")
    parser.add_argument('--num_trials', type=int, default=2, \
        help="Number of trials to run for each delta.")
    parser.add_argument("--save_path", type=str, default="./hpc_results/", 
        help="Path to save result files.")
    args = parser.parse_args()

    print(f"========= [1/5] setting up params =========")
    # if you change the list below, change the slurm submission script too
    δ_arr = np.linspace(0.5, 3.5, args.num_delta)
    # GW22 needs n > p TODO: mark the regime that charcoal fails
    # charcoal fails for δ < 1 
    δ = δ_arr[args.delta_idx]
    assert args.delta_idx < len(δ_arr), "delta_idx is out of range"
    print(f"δ = {δ}, δ_idx = {args.delta_idx}")
    p = args.p 
    n = int(δ * p) 
    σ = args.sigma

    # σ = 0.2 # noise standard deviation, differs from σ_w. 
    # noise std for charcoal is σ. noise std for AMP is σ/√n 
    # For some reason, too smal σ causes AMP to diverge for larger δ.

    Lmax = args.Lmax # L signals, L-1 changepoints
    L = args.L # ground truth number of signals
    assert L <= Lmax
    Lmin = 1
    print(f"Lmin = {Lmin}, Lmax = {Lmax}, L = {L}")
    var_β_1_fixed = 8
    σ_w_fixed = args.sigma_w # perturbation of β_2 from β_1, differs from σ

    α = 0.5 # fraction of regression coeff entries that differ
    k = α * p + np.sqrt(p) # upper bound on sparsity level rather than exact sparsity level

    ##### scale signal, noise so that SNR is constant #####
    var_β_1 = var_β_1_fixed * δ 
    σ_w = σ_w_fixed * np.sqrt(δ)
    #######################################################
    # var_β_1 = var_β_1_fixed 
    # σ_w = σ_w_fixed
    ρ_1 = var_β_1/ δ
    print(f'SNR = {1/n * p * var_β_1/σ**2}')
    η = np.sqrt(var_β_1/ (var_β_1 + σ_w**2))
    print(f'η = {η}')
    
    # In fact rho is not needed as an inpput to charcoal, but indicates how
    # challenging the detection problem is for charcoal:
    expected_l2_norm = α*n * ((1-η)**2 * var_β_1 + η**2 * σ_w**2)
    rho = np.sqrt(expected_l2_norm)
    # rho: signal strength at each changepoint, i.e. the l2 norm of each
    # change vector, if rho is of length 1, then the signal strength is applied to
    # all changes
  
    true_signal_prior = SparseDiffSignal(var_β_1, σ_w, α, Lmax) # var_1 = δρ_1
    ρ = jnp.array(true_signal_prior.cov/δ)
    print(f"ρ = {ρ} (CHECK THIS. If close to flat, then signals are too correlated to be distinguished)")
    
    amp_prep_start = time.time()
    if args.uniform_p_l:
        print("Using uniform prior over number of signals L, not signal configs.")
        num_L = args.Lmax - Lmin + 1
        p_l = np.ones((num_L, ))/num_L # uniform prior on L
    else:
        print("Using uniform prior over signal configs, not number of signals L.")
        p_l = None # uniform prior over configs
    est_signal_prior = true_signal_prior
    ρ_est = jnp.array(est_signal_prior.cov/δ)
    
    Δ = int(n * args.frac_Delta) # This needs to be small enough to capture true chgpts
    print(f"Δ = {Δ}")
    num_valid_configs, η_arr, p_η_arr, ϕ = unif_prior_to_η_φ(
        Lmin, Lmax, Δ=Δ, n=n, p_l=p_l) # Lxn matrix
    amp_prep_runtime = time.time() - amp_prep_start

    if L == 2:
        true_chgpt_locations = lambda n: [int(n/3)] 
    elif L == 3:
        true_chgpt_locations = lambda n: [int(n/3), int(8*n/15)]
    elif L == 4:
        true_chgpt_locations = lambda n: [int(n/3), int(8*n/15), int(11*n/15)]
    else:
        assert False, "true_chgpt_locations only defined for L=2,3,4"
    η_true = np.array(true_chgpt_locations(n))
    print(f"True chgpts = {η_true}")
    assert np.min(η_true) >= Δ and np.max(η_true) <= n - Δ, \
        "true chgpts too close to the boundary, not covered by prior"
    ψ_true = η_to_ψ(η_true, n)

    T = 15 # num AMP iterations
    # TODO: the following is only meaningful if AMP knows a priori theres exactly one chgpt.
    # hausdorff_rand_guess = 1/n * 1/2 * \
    #     (n - 2 * cp_true.item() + cp_true.item()**2 + (n-cp_true.item()) ** 2) # TODO: check +/-1s
    # norm_hausdorff_rand_guess = hausdorff_rand_guess / n
    # Fixed C_true for all trials.
    amp_hausdorff = -np.ones(args.num_trials)
    se_fixed_C_hausdorff = -np.ones(args.num_trials)
    charcoal_hausdorff = -np.ones(args.num_trials)

    amp_runtime = -np.ones(args.num_trials)
    charcoal_runtime = -np.ones(args.num_trials)
    # hausdorff_rand_guess = 0 # TODO: for 1 chgpt specifically
    num_runs_for_fail_safe = 2 # if any of the runs on competing algs fail, run up to this many times
    for i in tqdm(range(args.num_trials)):
        B̃ = true_signal_prior.sample(p)
        X = nprandom.normal(0, jnp.sqrt(1/n), (n, p))
        Θ = X @ B̃
        assert Θ.shape == (n, Lmax)
        Y = q(Θ, ψ_true, σ)
        assert Y.shape == (n, 1)

        if True:
            print('========== [2/5] Running AMP ==========')
            amp_start = time.time()
            # B̂, Θ_t_arr, ν_arr, κ_T_arr, ν̂_arr, κ_B_arr \
            #     = GAMP(δ, p, ϕ, σ, X, Y, T, \
            #     true_signal_prior=true_signal_prior, est_signal_prior=est_signal_prior)
            B̂, Θ_t_arr, ν_arr, κ_T_arr, ν̂_arr, κ_B_arr \
                = GAMP_full(B̃, δ, p, ϕ, σ, X, Y, ψ_true, T, \
                true_signal_prior=true_signal_prior, est_signal_prior=est_signal_prior)
            # Using avg version of ν and κ_T (along with σ, ρ, C_s: all possible chgpt configs) 
            # to define the posterior, and then evaluate the posterior at the actual observation
            # Y and AMP output Θ_t
            print("Calculating posterior...")
            post = posterior_over_η(η_arr, p_η_arr, Θ_t_arr[-1], Y, ρ_est, σ, ν_arr[-1], κ_T_arr[-1])
            assert post.shape == (num_valid_configs, )
            # post = posterior.compute_posterior(C_s, Θ_t, Y, n, ρ, σ, ν_arr[-1], κ_T_arr[-1])
            if False:
                plt.figure()
                plt.plot(post)
                plt.title("Posterior AMP")
                plt.xlabel("C chgpt config")
                plt.show()
            MAP_η_ = MAP_η(η_arr, post) # length-(L-1) vector
            print("MAP_η_: ", MAP_η_)
            AMP_chgpts = MAP_η_[MAP_η_ != -1]
            print("AMP estimated MAP chgpts (0-indexed, " + \
                f"starting location of new signals): {AMP_chgpts}")
            AMP_num_chgpts = len(AMP_chgpts)
            print(f"AMP num chgpts = {AMP_num_chgpts}")
            # MAP_C = posterior.MAP(C_s, post) # length-n vector storing signal index for each time point 
            # MAP_chgpt_vec = signal_configuration.C_to_chgpt(MAP_C) # changepoint locations
            # if MAP_chgpt_vec is None:
            #     print("MAP chgpt vec is None. Lets take a random guess:")
            #     amp_hausdorff[i] = norm_hausdorff_rand_guess
            # else:
            #     print("AMP estimated MAP chgpts: ", MAP_chgpt_vec)
            #     AMP_num_chgpts = MAP_chgpt_vec.shape[0]
            #     print(f"AMP num chgpts = {AMP_num_chgpts}") 
            #     amp_hausdorff[i] = 1/n * hausdorff(cp_true, MAP_chgpt_vec)
            amp_end = time.time()
            amp_runtime[i] = (amp_end - amp_start) + amp_prep_runtime
            amp_hausdorff[i] = 1/n * hausdorff(η_true, AMP_chgpts)

            ###### Run SE (with fixed C_true) ######
            print('========== [3/5] Running SE with fixed C ==========')
            # For smaller n, p AMP might not match SE; SE gives the limiting performance
            # Recall (Θ_t, Y) of fixed C AMP converges to (V_θt, q(Z_B, C_true)) in distribution
            # V_θt should be defined using ν_fixed, κ_T_fixed because the corresponding AMP
            # used fixed C_true
            ν_fixed_arr, κ_T_fixed_arr = SE_fixed_C_v1(
                ψ_true, true_signal_prior, est_signal_prior, δ, p, ϕ, Lmax, σ, T, \
                    ν_arr, κ_T_arr, ν̂_arr, κ_B_arr, tqdm_disable = True)
            ν_fixed = ν_fixed_arr[-1]
            κ_T_fixed = κ_T_fixed_arr[-1]
        
            # Sample according to fixed-C parameters:
            # V_θ follows the same distribution as Θ^t output by AMP, 
            # Y_bar follows the same distribution as Y:
        
            # κ_T_fixed = (κ_T_fixed + κ_T_fixed.T) / 2
            Z_B = jnp.array(np.random.multivariate_normal(np.zeros(Lmax), ρ, size=n))
            Y_bar = q(Z_B, ψ_true, σ)
            V_θ = Z_B @ jnp.linalg.inv(ρ) @ ν_fixed + jnp.array(np.random.multivariate_normal(
                np.zeros(Lmax), κ_T_fixed, size=n))

            post_se = posterior_over_η(η_arr, p_η_arr, V_θ, Y_bar, ρ, σ, ν_arr[-1], κ_T_arr[-1])
            # post_se = posterior.compute_posterior(C_s, V_θ, Y_bar, n, ρ, σ, ν_arr[-1], κ_T_arr[-1])
            if False:
                plt.figure()
                plt.plot(post_se)
                plt.title("Posterior SE")
                plt.xlabel("C chgpt config")
                plt.show()

            MAP_η_se_ = MAP_η(η_arr, post_se) # length-(L-1) vector
            print("MAP_η_se_: ", MAP_η_se_)
            se_chgpts = MAP_η_se_[MAP_η_se_ != -1]
            print("SE fixed C estimated MAP chgpts (0-indexed, " + \
                f"starting location of new signals): {se_chgpts}")
            se_num_chgpts = len(se_chgpts)
            print(f"se num chgpts = {se_num_chgpts}")

            # MAP_C_se = posterior.MAP(C_s, post_se) 
            # MAP_chgpt_vec_se = signal_configuration.C_to_chgpt(MAP_C_se)
            # Avoid empty set which would lead to infinite Hausdorff distance.
            # Match baseline random guess:
            # if MAP_chgpt_vec_se is None: 
            #     print("MAP chgpt vec_se is None. Lets take a random guess:")
            #     se_fixed_C_hausdorff[i] = norm_hausdorff_rand_guess
            # else:
            #     print("SE estimated MAP chgpts: ", MAP_chgpt_vec_se)
            #     num_chgpts_se = MAP_chgpt_vec_se.shape[0]
            #     print(f"SE num chgpts = {num_chgpts_se}")
            #     se_fixed_C_hausdorff[i] = 1/n * hausdorff(cp_true, MAP_chgpt_vec_se)
            se_fixed_C_hausdorff[i] = 1/n * hausdorff(η_true, se_chgpts)

        if False:
            print('========== [4/5] Running charcoal ==========')
            # Transform the data so that X ~ N(0, 1)
            X_unit = np.array(np.sqrt(n) * X)
            Y_unit = np.array(np.sqrt(n) * Y)
            for j in range(num_runs_for_fail_safe):
                try:
                    charcoal_start = time.time()
                    charcoal_hausdorff[i] = run_charcoal(X_unit, Y_unit, σ, η_true)
                    charcoal_end = time.time()
                    charcoal_runtime[i] = charcoal_end - charcoal_start
                    print(f"charcoal finished in {charcoal_runtime[i]} seconds")
                    print(f"trial {j} succeeded")
                    break
                except:
                    print(f"trial {j} failed")
            if charcoal_hausdorff[i] == -1 or charcoal_runtime[i] == -1:
                print(f"charcoal failed all {num_runs_for_fail_safe} trials, saving nans")
                charcoal_hausdorff[i] = np.nan
                charcoal_runtime[i] = np.nan

    print(f"========= [5/5] saving results to file =========")
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = "chgpts_sparse_diff_sigma_w_" + '{0:.2f}'.format(args.sigma_w) + \
        "_delta_idx_" + "{:02d}".format(args.delta_idx) + "_" + timestamp
    # log_file_name = args.save_path + file_name + '.log'
    data_file_name = args.save_path + file_name + '.npz'    

    np.savez(data_file_name, p=p, α=α, k=k, n=n, δ=δ, σ=σ, σ_w=σ_w, 
        var_β_1=var_β_1, ρ_1=ρ_1, Lmin=Lmin, Lmax=Lmax, L=L, T=T, num_trials=args.num_trials, 
        rho=rho, var_β_1_fixed=var_β_1_fixed, σ_w_fixed=σ_w_fixed,
        amp_hausdorff=amp_hausdorff,  
        se_hausdorff=se_fixed_C_hausdorff, 
        charcoal_hausdorff=charcoal_hausdorff, 
        amp_runtime=amp_runtime, charcoal_runtime=charcoal_runtime,
        delta_idx=args.delta_idx, uniform_p_l=args.uniform_p_l, η_true=η_true, 
        frac_Delta=args.frac_Delta)
    print(f"========= results saved to file =========")