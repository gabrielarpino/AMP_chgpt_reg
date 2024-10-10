import numpy as np
# from . import ϵ_0
from jax.scipy.special import logsumexp
import jax.numpy as jnp
import jax
from amp.covariances import Σ_V_j_Y_j
from jax.scipy.stats import multivariate_normal
import amp.marginal_separable_jax

# Key challenge lies in computing the likelihood. 

def L(V, u, ψ, true_signal_prior, est_signal_prior,
        δ, p, ϕ_, L, σ, T, \
        ν_avg_arr, κ_T_avg_arr, ν̂_avg_arr, κ_B_avg_arr, st_ζ = None, tqdm_disable = False): 

    ν_fixed_arr, κ_T_fixed_arr = amp.marginal_separable_jax.SE_fixed_C_v1(ψ, true_signal_prior, est_signal_prior, \
                                    δ, p, ϕ_, L, σ, T, \
                                    ν_avg_arr, κ_T_avg_arr, ν̂_avg_arr, κ_B_avg_arr, st_ζ = None, tqdm_disable = False)


    

    return