import argparse
import datetime
import time
import rpy2
from amp.comparison import DCDP, DP, DPDU
from amp.posterior import MAP_η, posterior_over_η, η_to_ψ

from amp.signal_priors import SparseGaussianSignal
# utils = importr('utils')
# base = importr('base')
# chgpts = importr('changepoints')
# charcoal = importr('charcoal')
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import jax.numpy as jnp
import numpy.random as nprandom
import numpy as np
import tqdm
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)
import matplotlib.pyplot as plt
from amp import hausdorff
from amp.marginal_separable_jax import SE_fixed_C_v1, q, GAMP_full
from amp.signal_configuration import unif_prior_to_η_φ

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        'Run DP, DPDU, DCDP and AMP on sparse signals for" + \
            " varying oversampling ratio delta.')
    parser.add_argument('--num_delta', type=int, default=10,
        help="Number of delta values or number of jobs in job array.")
    parser.add_argument('--delta_idx', type=int, default=0,
        help="Index of delta in the array of delta values to use.")
    parser.add_argument('--p', type=int, default=200, help="Number of covariates.")
    parser.add_argument('--sigma', type=float, default=0.1, help="noise std.")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="sparsity level. fraction of nonzeros")
    parser.add_argument('--L', type=int, default=3, help="ground truth number of signals.")
    parser.add_argument('--Lmax', type=int, default=4, help="max number of signals.")
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
    
    print(f"========= [1/7] setting up params =========")
    δ_arr = np.linspace(0.5, 3.5, args.num_delta)
    δ = δ_arr[args.delta_idx]
    assert args.delta_idx < len(δ_arr), "delta_idx is out of range"
    print(f"δ = {δ}, δ_idx = {args.delta_idx}")
    p = args.p # 300
    n = int(δ * p)

    σ = args.sigma # noise standard deviation
    Lmax = args.Lmax
    L = args.L # ground truth number of signals
    Lmin = 1
    print(f"Lmin = {Lmin}, Lmax = {Lmax}, L = {L}")
    α = args.alpha # sparsity parameter, fraction of nonzeros. 
    σ_l_arr_unscaled = np.ones((Lmax, )) # signal standard deviation
    # Rescale σ_l_arr_unscaled so that SNR = p/n * σ_l_arr**2 / σ2 is fixed for different δ
    σ_l_arr = σ_l_arr_unscaled * np.sqrt(δ)
    # assert np.allclose(p/n * σ_l_arr**2, σ_l_arr_unscaled**2) 
    # Above can fail because n is integer of δ*p.
    print(f"signal power = {p/n * σ_l_arr**2} CHECK THIS, it should be close to {σ_l_arr_unscaled**2}")
    true_signal_prior = SparseGaussianSignal(α, δ, σ_l_arr)
    ρ = jnp.array(true_signal_prior.cov/δ)
    
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

    # min distance between chgpts: 
    # Δ = int(n/10)
    Δ = int(n * args.frac_Delta)
    # p_l = np.ones(num_L) / num_L # uniform prior on L
    # TODO: include the time for this initialisation step into AMP itself:
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
    η_true_extended = np.concatenate((η_true, -np.ones(Lmax-L)))
    print(f"True chgpts = {η_true}")
    assert np.min(η_true) >= Δ and np.max(η_true) <= n - Δ, \
        "true chgpts too close to the boundary, not covered by prior"
    ψ_true = η_to_ψ(np.array(η_true), n)

    T = 15 # AMP iterations
    amp_hausdorff = -np.ones(args.num_trials)
    se_fixed_C_hausdorff = -np.ones(args.num_trials)
    DPDU_hausdorff = -np.ones(args.num_trials)
    DP_hausdorff = -np.ones(args.num_trials)
    DCDP_hausdorff = -np.ones(args.num_trials)

    amp_runtime = -np.ones(args.num_trials)
    DPDU_runtime = -np.ones(args.num_trials)
    DP_runtime = -np.ones(args.num_trials)
    DCDP_runtime = -np.ones(args.num_trials)
    num_runs_for_fail_safe = 2 # if any of the runs on competing algs fail, run up to this many times
    for i in tqdm(range(args.num_trials)):
        B̃ = true_signal_prior.sample(p)
        X = nprandom.normal(0, np.sqrt(1/n), (n, p))
        Θ = X @ B̃
        Y = q(Θ, ψ_true, σ) 
        # Y = q(Θ, ψ_true, σ/np.sqrt(n)) 
        # Note, dividing σ by √n here so that when we renormalize 
        # the X and Y after, and get X ~ N(0, 1) and ϵ ~ N(0, σ²). 

        if True:
            print("===================== [2/7] Running AMP =====================")
            amp_start = time.time()
            # B̂, Θ_t_arr, ν_arr, κ_T_arr, ν̂_arr, κ_B_arr \
            #     = GAMP(δ, p, ϕ, σ, X, Y, T, 
            #     true_signal_prior=true_signal_prior, est_signal_prior=est_signal_prior)
            B̂, Θ_t_arr, ν_arr, κ_T_arr, ν̂_arr, κ_B_arr \
                = GAMP_full(B̃, δ, p, ϕ, σ, X, Y, ψ_true, T, 
                true_signal_prior=true_signal_prior, est_signal_prior=est_signal_prior)

            print("Calculating posterior...")
            post = posterior_over_η(η_arr, p_η_arr, Θ_t_arr[-1], Y, ρ_est, σ, ν_arr[-1], κ_T_arr[-1])
            assert post.shape == (num_valid_configs, )
            MAP_η_ = MAP_η(η_arr, post) # length-(L-1) vector
            print("MAP_η_: ", MAP_η_)
            AMP_chgpts = MAP_η_[MAP_η_ != -1]
            print("AMP estimated MAP chgpts (0-indexed, " + \
                f"starting location of new signals): {AMP_chgpts}")
            AMP_num_chgpts = len(AMP_chgpts)
            print(f"AMP num chgpts = {AMP_num_chgpts}")
            amp_runtime[i] = (time.time() - amp_start) + amp_prep_runtime
            amp_hausdorff[i] = 1/n * hausdorff(η_true, AMP_chgpts)
            print(f"========== finished in {amp_runtime[i]} seconds ==========")

            print('========== [3/7] Running SE with fixed C ==========')
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
            se_fixed_C_hausdorff[i] = 1/n * hausdorff(η_true, se_chgpts)

        if False:
            # Transform the data so that X ~ N(0, 1)
            X_unit = np.array(np.sqrt(n) * X)
            Y_unit = np.array(np.sqrt(n) * Y)

            print("================= [4/7] Running DPDU =================")
            for j in range(num_runs_for_fail_safe):
                try:
                    DPDU_start = time.time()
                    DPDU_chgpt_est = DPDU(X_unit, Y_unit)
                    print("DPDU estimated chgpts: ", DPDU_chgpt_est)
                    DPDU_runtime[i] = time.time() - DPDU_start
                    DPDU_hausdorff[i] = 1/n * hausdorff(DPDU_chgpt_est, η_true)
                    print(f"========== finished in {DPDU_runtime[i]} seconds ==========")
                    print(f"Trial {j} succeeded")
                    break
                except:
                    print(f"DPDU trial {j} failed")
            if DPDU_runtime[i] == -1 or DPDU_hausdorff[i] == -1:
                print(f"DPDU failed in all {num_runs_for_fail_safe} trials. Saving nans.")
                DPDU_runtime[i] = np.nan
                DPDU_hausdorff[i] = np.nan

            print("====================== [5/7] Running DP ======================")
            λs = [0.1, 0.5, 1, 2, 4] # Too small gives numerical errors
            γs = [0.1, 1, 2, 4, 8, 16]
            for j in range(num_runs_for_fail_safe):
                try:
                    DP_start = time.time()
                    # 3rd input is max number of changepoints:
                    DP_chgpt_est = DP(X_unit, Y_unit, Lmax-1, λs, γs)
                    print("DP estimated chgpts: ", DP_chgpt_est)
                    DP_runtime[i] = time.time() - DP_start
                    DP_hausdorff[i] = 1/n * hausdorff(DP_chgpt_est, η_true)
                    print(f"========== finished in {DP_runtime[i]} seconds ==========")
                    print(f"Trial {j} succeeded")
                    break
                except:
                    print(f"DP trial {j} failed")
            if DP_runtime[i] == -1 or DP_hausdorff[i] == -1:
                print(f"DP failed in all {num_runs_for_fail_safe} trials. Saving nans.")
                DP_runtime[i] = np.nan
                DP_hausdorff[i] = np.nan

            print("===================== [6/7] Running DCDP =====================")
            # This is the most stable, and is the fastest to run.
            for j in range(num_runs_for_fail_safe):
                try:
                    DCDP_start = time.time()
                    DCDP_chgpt_est = DCDP(X_unit, Y_unit)
                    print("DCDP estimated chgpts: ", DCDP_chgpt_est)
                    DCDP_runtime[i] = time.time() - DCDP_start
                    DCDP_hausdorff[i] = 1/n * hausdorff(DCDP_chgpt_est, η_true)
                    print(f"========== finished in {DCDP_runtime[i]} seconds ==========")
                    print(f"Trial {j} succeeded")
                    break
                except:
                    print(f"DCDP trial {j} failed")
            if DCDP_runtime[i] == -1 or DCDP_hausdorff[i] == -1:
                print(f"DCDP failed in all {num_runs_for_fail_safe} trials. Saving nans.")
                DCDP_runtime[i] = np.nan
                DCDP_hausdorff[i] = np.nan
        
    print("========================= [7/7] Saving results ===============================")
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = "chgpts_sparse_compare" + \
        "_delta_idx_" + "{:02d}".format(args.delta_idx) + "_" + timestamp
    data_file_name = args.save_path + file_name + '.npz'    

    np.savez(data_file_name, delta_idx=args.delta_idx, σ=σ, L=L, δ=δ,
            Lmax=args.Lmax, Lmin=Lmin, α=α, σ_l_arr=σ_l_arr, p=p, n=n,
            T=T, Δ=Δ, p_l=p_l,
            num_trials=args.num_trials, amp_hausdorff=amp_hausdorff,
            se_fixed_C_hausdorff=se_fixed_C_hausdorff,
            DPDU_hausdorff=DPDU_hausdorff, DP_hausdorff=DP_hausdorff,
            DCDP_hausdorff=DCDP_hausdorff,
            amp_runtime=amp_runtime, DPDU_runtime=DPDU_runtime,
            DP_runtime=DP_runtime, DCDP_runtime=DCDP_runtime,
            η_true=η_true, frac_Delta = args.frac_Delta)
    print("========= Reached the end of script =========")