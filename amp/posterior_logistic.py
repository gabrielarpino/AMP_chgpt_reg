import numpy as np
# from . import ϵ_0
from jax.scipy.special import logsumexp
import jax.numpy as jnp
import jax
from amp.covariances import Σ_V_j_Y_j
from jax.scipy.stats import multivariate_normal, norm
from amp.signal_configuration import η_to_ψ, η_to_ψ_jax
import amp.marginal_separable_jax_logistic
from tqdm.auto import tqdm
from jax_tqdm import scan_tqdm

# New version converts individual η (length-(L-1) vector) into Ψ 
# or C ((length-n) vector),
# η_arr is a num_configs x (L-1) matrix, where each row stores the 
# starting index of signal 1 up to signal (L-1) (signal 0 always 
# starts at index 0 so isnt stored).

# Old version directly uses all possible Ψ (i.e. a large 
# num_configs x n matrix called C_s) which is memory intensive.

def tabulate_state_evolution(η_arr, true_signal_prior, est_signal_prior, \
                                    δ, p, ϕ_, L, σ, T, \
                                    ν_avg_arr, κ_T_avg_arr, ν̂_avg_arr, κ_B_avg_arr, st_ζ = None, tqdm_disable = False):
    # Tabulate the ν, κ_T for each possible signal configuration
    ν_fixed_list = []
    κ_T_fixed_list = []
    n = ϕ_.shape[1]

    def SE_fixed_C_clean(ψ):
        return amp.marginal_separable_jax_logistic.SE_fixed_C_v1(ψ, true_signal_prior, est_signal_prior, \
                                    δ, p, ϕ_, L, σ, T, \
                                    ν_avg_arr, κ_T_avg_arr, ν̂_avg_arr, κ_B_avg_arr, st_ζ = st_ζ, tqdm_disable = tqdm_disable)
    SE_fixed_C_jit = jax.jit(SE_fixed_C_clean)
    # ψ_s = η_to_ψ_jax_combinations_mapped(η_arr, n, L)

    for i in tqdm(range(len(η_arr))):  # TODO, use JAX or jit to speed this up
        η_candidate = η_arr[i]
        ψ = η_to_ψ_jax(η_candidate, ϕ_.shape[1])
        # ψ = ψ_s[i]
        # ν_fixed_arr, κ_T_fixed_arr = amp.marginal_separable_jax.SE_fixed_C_v1(ψ, true_signal_prior, est_signal_prior, \
        #                             δ, p, ϕ_, L, σ, T, \
        #                             ν_avg_arr, κ_T_avg_arr, ν̂_avg_arr, κ_B_avg_arr, st_ζ = None, tqdm_disable = False)
        ν_fixed_arr, κ_T_fixed_arr = SE_fixed_C_jit(ψ)
        ν_fixed_list.append(ν_fixed_arr[-1])
        κ_T_fixed_list.append(κ_T_fixed_arr[-1])
    ν_fixed_tabulated = jnp.array(ν_fixed_list).reshape(len(η_arr), L, L)
    κ_T_fixed_tabulated = jnp.array(κ_T_fixed_list).reshape(len(η_arr), L, L) 

    return ν_fixed_tabulated, κ_T_fixed_tabulated

def exact_posterior(η_arr, p_η_arr, V, Y, ρ, σ, ν_fixed_tabulated, κ_T_fixed_tabulated):

    a = logP_V_Y_given_η_jax_allη(η_arr, V, Y, ρ, σ, ν_fixed_tabulated, κ_T_fixed_tabulated).reshape(-1, 1)
    a_max = jnp.max(a)
    b = p_η_arr.reshape(-1, 1)
    log_num, sgn_log_num = logsumexp(a=a-a_max, axis=1, b=b, return_sign=True)
    log_denom, sgn_log_denom = logsumexp(a=a-a_max, axis=0, b=b, return_sign=True)
    posterior = jnp.exp(sgn_log_num*log_num - sgn_log_denom*log_denom) # length num_configs vector 
    assert jnp.isclose(jnp.sum(posterior), 1)
    return posterior

def logP_V_Y_given_η_jax(η, V, Y, ρ, σ, ν, κ_T):
    """ TODO: This uses η_to_ψ_jax, which assumes η has to -1's in it. Need to change η_to_ψ_jax to fix this. """
    n = Y.shape[0]
    ψ = η_to_ψ_jax(η, n) # length-n vector
    # log P(V, Y | η) = log ∑_j P(Vj, Yj | ψj)
    # sum over all rows of V, Y, ψ:
    return jnp.sum(logP_Vj_Yj_given_ψj_allj(V, Y, ψ.reshape(-1,1), ρ, σ, ν, κ_T)) # scalar

def logP_V_Y_given_η_jax_allη(η_arr, V, Y, ρ, σ, ν_list, κ_T_list):
    return jax.vmap(logP_V_Y_given_η_jax, (0, None, None, None, None, 0, 0), 0)(η_arr, V, Y, ρ, σ, ν_list, κ_T_list)


### --- Functions from the approximate posterior --- ###


def logP_Vj_Yj_given_ψj(Vj, Yj, ψj, ρ, σ, ν, κ_T):
    """
    Calculates log P(Vj, Yj | ψj).
    
    Vj ∈ R^L is the j-th row of V.
    Yj ∈ R is the j-th entry of q(Z, ψ, ϵ).
    ψj ∈ R is the j-th entry of ψ.
    """
    L = ρ.shape[0]
    lam = np.sqrt(np.pi/8)

    Σ = ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T     # covariance matrix of V
    Σ_Z_given_V = (ρ - ν.T @ jnp.linalg.inv(Σ) @ ν)
    μ_s = (ν @ jnp.linalg.inv(Σ) @ (Vj.reshape(L,1).flatten()))[ψj]

    σ_sq_s = Σ_Z_given_V.diagonal().flatten()[ψj]

    ρ_s = lam/jnp.sqrt(1+lam**2*σ_sq_s)

    res_0 = norm.logcdf(-ρ_s*μ_s, loc=0, scale=1)
    res_1 = norm.logcdf(ρ_s*μ_s, loc=0, scale=1)

    res =  (Yj==0).astype(int)*res_0 + (Yj==1).astype(int)*res_1
    return res # scalar

# logP_Vj_Yj_given_ψj_allj = jax.jit(jax.vmap(logP_Vj_Yj_given_ψj, (0, 0, 0, None, None, None, None), 0)) # length-n vector
# If Vmap is too memory intensive here, use map. 
# logP_Vj_Yj_given_ψj_allj = jax.jit(jax.lax.map(logP_Vj_Yj_given_ψj, (0, 0, 0, None, None, None, None), 0)) # length-n vector
def logP_Vj_Yj_given_ψj_allj(V, Y, ψ, ρ, σ, ν, κ_T):

    L = V.shape[1]
    n = V.shape[0]
    def helper(x):
        V = x[:L]
        Y = x[L]
        ψ = x[L+1].astype(int)
        return logP_Vj_Yj_given_ψj(V, Y, ψ, ρ, σ, ν, κ_T)

    x = jnp.column_stack((V, Y, ψ))
    return jax.lax.map(helper, x)

def logP_V_Y_given_η(η, V, Y, ρ, σ, ν, κ_T):
    """
    Calculates log P(V, Y | η) = log P(V, Y | ψ) where ψ = η_to_ψ(η).
    η here is length-(L-1) vector
    """
    n = Y.shape[0]
    ψ = η_to_ψ(η, n) # length-n vector
    # log P(V, Y | η) = log ∑_j P(Vj, Yj | ψj)
    # sum over all rows of V, Y, ψ:
    return jnp.sum(logP_Vj_Yj_given_ψj_allj(V, Y, ψ.reshape(-1,1),
        ρ, σ, ν, κ_T)) # scalar

def logP_V_Y_given_η_allη(η_arr, V, Y, ρ, σ, ν, κ_T):
    nconfigs = η_arr.shape[0]
    logP_allη = np.zeros(nconfigs)
    for i_config in range(nconfigs): # Slow for loop
        η = η_arr[i_config, :]
        logP_allη[i_config] = logP_V_Y_given_η(η, V, Y, ρ, σ, ν, κ_T)
    # logP_allη = jax.vmap(logP_V_Y_given_η, (0, None, None, None, None, None, None), 0)(η_arr, V, Y, ρ, σ, ν, κ_T)
    return logP_allη

# The following doesnt work due to η_to_ψ involving dynamically sized arrays:
# logP_V_Y_given_η_allη = jax.vmap(logP_V_Y_given_η, 
#     (0, None, None, None, None, None, None), 0) # length-num_configs vector

def posterior_over_η(η_arr, p_η_arr, V, Y, ρ, σ, ν, κ_T):
    """
    η_arr is num_configs x (L-1) matrix.
    p_η_arr is length-num_configs vector, normalised to 1,
    storing the prior over η_arr.

    Returns posterior over η_arr, a length-num_configs vector.
    """
    a = logP_V_Y_given_η_allη(η_arr, V, Y, ρ, σ, ν, κ_T).reshape(-1, 1)
    a_max = jnp.max(a)
    b = p_η_arr.reshape(-1, 1)
    log_num, sgn_log_num = logsumexp(a=a-a_max, axis=1, b=b, return_sign=True)
    log_denom, sgn_log_denom = logsumexp(a=a-a_max, axis=0, b=b, return_sign=True)
    posterior = jnp.exp(sgn_log_num*log_num - sgn_log_denom*log_denom) # length num_configs vector 
    assert jnp.isclose(jnp.sum(posterior), 1)
    return posterior
 
def MAP_η(η_arr, posterior):
    """
    η_arr is num_configs x (L-1) matrix.
    posterior is length-num_configs vector.
    """
    assert η_arr.shape[0] == len(posterior)
    return η_arr[jnp.argmax(posterior), :]

def greedy_ML_η(η_arr, post, Δ_n):
    """ Search the posterior for the MAP estimate, but start by finding the most likely single changepoint, 
    and then conditioning on that and finding the next most likely, and so on. 
    
    C_s: num_configs x n array of all possible configurations

    Assume that the first signal is L = 1
    Assume that the next n - 2*Δ(n) signals are L = 2 (Δ(n) is the mandatory distance from the endpoints)
    Assume that the rest are L = 3

    Currently only works for L_min = 1, L_max = 3. 

    This greedy algorithm can only be made to run in linear time if the posterior is uniform (i.e. max likelihood), because then we don't have to compute the denominator of the posterior. 
    This assumes the prior is uniform (but can have Δ(n)'s etc. ).
    """

    num_configs = η_arr.shape[0]
    n = η_arr.shape[1]

    # Zeroth chgpt
    zeroth_chgpt_post = post[0]
    zeroth_chgpt_ML_idx = 0
    zeroth_chgpt_ML_max = zeroth_chgpt_post
    zeroth_chgpt_ML_η = η_arr[zeroth_chgpt_ML_idx, :]

    # Find the MAP estimate for the first signal
    first_chgpt_post = post[1:n - 2*Δ_n]
    first_chgpt_ML_idx = np.argmax(first_chgpt_post)
    first_chgpt_location = η_arr[first_chgpt_ML_idx, 0]
    first_chgpt_ML_max = post[first_chgpt_ML_idx]
    first_chgpt_ML_η = η_arr[first_chgpt_ML_idx]

    # Find the MAP estimate for the second signal, but only considering the configurations that have the first signal
    second_chgpt_post = post[n - 2*Δ_n:]
    candidate_list = []
    for η_idx in range(n-2*Δ_n, num_configs):
        if η_arr[η_idx, 0] == first_chgpt_location:
            candidate_list.append(η_idx)
    second_chgpt_post = second_chgpt_post[np.array(candidate_list)]
    second_chgpt_ML_idx = np.argmax(second_chgpt_post) # This is not implemented in linear time, but it can be if we construct the configurations from scratch in this function and evaluate the likelihood on individual configs.
    second_chgpt_ML_max = second_chgpt_post[second_chgpt_ML_idx]
    second_chgpt_ML_η = η_arr[second_chgpt_ML_idx]

    # Return the index of the maximum of the above three
    max_idx = np.argmax([zeroth_chgpt_ML_max, first_chgpt_ML_max, second_chgpt_ML_max])
    max_ML_η = [zeroth_chgpt_ML_η, first_chgpt_ML_η, second_chgpt_ML_η][max_idx]

    return max_ML_η

def multiple_MAP_η(η_arr, posterior, num_ηs):
    """Return first num_ηs ηs carrying most posterior mass."""
    num_configs = η_arr.shape[0]
    assert num_configs == len(posterior)
    assert num_ηs <= num_configs
    indices = jnp.argsort(posterior)[::-1][:num_ηs]
    return η_arr[indices, :], posterior[indices]

def bayes_credible_set_η(L:int, α_CI : float, η_arr: jnp.array, post: jnp.array):
    assert L == 2, "CI only implemented for L=2"
    
    num_configs = η_arr.shape[0]
    assert num_configs == len(post), "η_array and post should have same length"
    cdf_lower = 0
    idx_lower = 0
    while cdf_lower < α_CI/2:
        cdf_lower += post[idx_lower]
        idx_lower += 1
    cdf_upper = 0
    idx_upper = num_configs-1
    while cdf_upper < α_CI/2:
        cdf_upper += post[idx_upper]
        idx_upper -= 1
    if idx_lower > idx_upper:
        idx_lower -= idx_lower
        # idx_upper += idx_upper
    assert idx_lower <= idx_upper, "Something went wrong with the CI"
    return idx_lower, idx_upper

def left_sided_credible_set_η(L:int, α_CI : float, η_arr: jnp.array, post: jnp.array):
    assert L == 2, "CI only implemented for L=2"
    
    prob_vector = post

    # Set the target sum for the centered set
    target_sum = 1 - α_CI

    # Find the left-sided set
    left_set = prob_vector
    cumulative_sum = 0
    idx = 0

    for i in range(len(left_set)):
        cumulative_sum += left_set[i]
        idx += 1
        
        # Check if the cumulative sum is greater than or equal to half of the target sum
        if cumulative_sum >= target_sum:
            # left_set[i] -= cumulative_sum - target_sum
            # idx -= 1
            break

    return (0, idx)

def left_sided_credible_set_C(L, α_CI, C_s, post):

    assert L == 2, "CI only implemented for L=2"
    
    prob_vector = post

    # Set the target sum for the centered set
    target_sum = 1 - α_CI

    # Find the left-sided set
    left_set = prob_vector
    cumulative_sum = 0
    idx = 0

    for i in range(len(left_set)):
        cumulative_sum += left_set[i]
        idx += 1
        
        # Check if the cumulative sum is greater than or equal to half of the target sum
        if cumulative_sum >= target_sum:
            # left_set[i] -= cumulative_sum - target_sum
            # idx -= 1
            break

    return (0, idx)
######################### OLD CODE #########################
# from . import ϵ_0
ϵ_0 = 1e-6

def log_P_V_Y_given_C(C, V, Y, n, ρ, σ, ν, κ_T):
    L = ν.shape[0]
    assert (Y.shape == (n, 1) or Y.shape == (n,)), "Y should be a vector"
    assert V.shape == (n, L)

    Y = Y.reshape(-1, 1) # make sure Y is a column vector
    return jnp.sum(logP_Vj_Yj_given_ψj_allj(V, Y, C, ρ, σ, ν, κ_T)) # sum over all rows of V, Y, C
log_lik_tensorized = jax.vmap(log_P_V_Y_given_C, (0, None, None, None, None, None, None, None), 0)

# Replace the above log_lik_tensorized function with a jax.scan version to save on memory, for a small speed cost.
def log_lik_tensorized_scan(C_s, V, Y, n, ρ, σ, ν, κ_T):
    @scan_tqdm(C_s.shape[0], tqdm_type='auto')
    @jax.jit
    def scan_fn(carry, x):
        # C = x
        C = C_s[x]
        return 0, log_P_V_Y_given_C(C, V, Y, n, ρ, σ, ν, κ_T)
    # _, log_lik_mapped = jax.lax.scan(scan_fn, 0, C_s)
    _, log_lik_mapped = jax.lax.scan(scan_fn, 0, jnp.arange(C_s.shape[0])) # Allows to run tqdm on scan
    return log_lik_mapped

def compute_approx_posterior(C_s, V, Y, n, ρ, σ, ν, κ_T):
    # Assuming p(C) is uniform over the columns in C_s and zero elsewhere.
    log_lik_mapped = log_lik_tensorized_scan(C_s, V, Y, n, ρ, σ, ν, κ_T)  # map over all C_s (C is one column of C_s). Scan version is less memory intensive. 
    log_Z_ = logsumexp(a=log_lik_mapped, axis=0, return_sign=False)
    return jnp.exp(log_lik_mapped - log_Z_) # returns a vector of length C_s.shape[1] (number of columns in C_s)

def comp_posterior_with_prior(C_s, V, Y, n, ρ, σ, ν, κ_T, prior):
    # Assuming p(C) is uniform over the columns in C_s and zero elsewhere.
    a = log_lik_tensorized(C_s, V, Y, n, ρ, σ, ν, κ_T).reshape(-1, 1)  # map over all C_s (C is one column of C_s)
    a_max = jnp.max(a)
    b = prior.reshape(-1, 1)
    log_num, sgn_log_num = logsumexp(a=a-a_max, axis=1, b=b, return_sign=True)
    log_denom, sgn_log_denom = logsumexp(a=a-a_max, axis=0, b=b, return_sign=True)
    posterior = jnp.exp(sgn_log_num*log_num - sgn_log_denom*log_denom) # length num_configs vector 
    # assert jnp.isclose(jnp.sum(posterior), 1)
    return posterior 

def MAP(C_s, post):
    max_idx = jnp.argmax(post)
    return C_s[max_idx]

def bayes_credible_set(L: int, α_CI: float, C_s: jnp.array, post: jnp.array):
    """
    Compute the Bayesian confidence interval for the changepoint locations.
    Since CI makes most sense pictorially for one chgpt, this function
    allows L=2 for now.

    C_s: num_configs x n array of all possible configurations
    The CI contains the ground truth with probability 1-α_CI.
    """
    assert L == 2, "CI only implemented for L=2"
    
    num_configs = C_s.shape[0]
    for i_config in range(num_configs):
        assert len(jnp.unique(C_s[i_config, :])) == L, \
        "C_s should contain configs containing exactly 1 chgpt"
    assert num_configs == len(post), "C_s and post should have same length"
    cdf_lower = 0
    idx_lower = 0
    while cdf_lower < α_CI/2:
        cdf_lower += post[idx_lower]
        idx_lower += 1
    cdf_upper = 0
    idx_upper = num_configs-1
    while cdf_upper < α_CI/2:
        cdf_upper += post[idx_upper]
        idx_upper -= 1
    if idx_lower > idx_upper:
        idx_lower = idx_upper
        idx_upper = idx_lower + 1
    assert idx_lower <= idx_upper, "Something went wrong with the CI"
    return idx_lower, idx_upper

# a = np.array([[0.92001345, 0.89247577, 0.92001345],
#  [0.89247577, 0.92102642, 0.89247577],
#  [0.92001345, 0.89247577, 0.67666667]])

# b = np.array([[0.98410817, 0.94955782, 0.98410817],
#     [0.94955782, 0.98383919, 0.94955782],
#     [0.98410817, 0.94955782, 0.34333333]])