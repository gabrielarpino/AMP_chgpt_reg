import numpy as np
from amp.posterior import η_to_ψ_jax
import jax
import jax.numpy as jnp
# Code for the goodness of fit criteria that are used to estimate the change point locations. 

def gof_linear(θ, y): 
    """ Negative log-likelihood for the linear model (normalized)."""
    # return (1 / y.shape[0]) * jnp.linalg.norm(y - θ) **2
    return (1 / y.shape[0]) * jnp.power(y - θ, 2)

def gof_logistic(θ, y): 
    """ Negative log-likelihood for the logistic model (normalized)."""
    return (1 / y.shape[0]) * (jnp.log(1 + jnp.exp(θ)) - jnp.multiply(y, θ))

def gof_rectified_linear(θ, y): 
    """ Negative log-likelihood for the rectified linear model (normalized)."""
    return (1 / y.shape[0]) * jnp.power(y - jnp.maximum(0, θ), 2)

def argmax_gof(η_arr, Θ_matrix, y, model = 'linear'): 
    """ Returns the argmax of the goodness of fit criteria, over all possible change points. Assumes that η_arr does not contain endpoints. Tested, works. """

    if model == 'linear': 
        gof = gof_linear
    elif model == 'logistic': 
        gof = gof_logistic
    elif model == 'rectified linear': 
        gof = gof_rectified_linear
    else: 
        raise ValueError('Model not recognized.')

    def gof_η(η, θ_, y): 
        """ Returns the goodness of fit for a given change point η. """
        num_chgpts = η.size
        ψ = η_to_ψ_jax(η, θ_.shape[0])

        if num_chgpts == 0:
            return gof(θ_, y)
        
        j_prev = 0
        res = 0
        # Do the first one separately
        ℓ = 0
        cond_ = (ψ == ℓ)
        gof_ = gof(θ_[:, ℓ].flatten(), y.flatten())
        where_ = jnp.where(ψ == ℓ, gof_, 0)
        res += where_.sum()

        if num_chgpts < 1:
            return res
        for j in range(1, num_chgpts + 1): 
            ℓ = j
            cond_ = (ψ == ℓ)
            gof_ = gof(θ_[:, ℓ].flatten(), y.flatten())
            where_ = jnp.where(ψ == ℓ, gof_, 0)
            res += where_.sum()
        return res
    gof_η_mapped = jax.jit(jax.vmap(gof_η, in_axes = (0, None, None)))

    gof_list = gof_η_mapped(η_arr, Θ_matrix, y)
    amin = jnp.argmin(gof_list)
    return η_arr[amin], gof_list

def lin_reg_opt_chgpt_gof_combinations(Z, y, η_arr, p_η_arr, σ = 0.5, ϵ = 0.15):
    """ Z is n × L, where L is the number of signals. 
    Don't need to vmap this whole thing, but can jit the inside of for loop to make it faster. 
    Assumes we are using the linear regression loss. """

    L = Z.shape[1]
    C_s, η_idx = η_to_ψ_jax_combinations_mapped(η_arr, Z.shape[0], L)

    # Implement with jax.lax.scan. Uses less memory than vmap, and nearly as fast.
    gof_list = []
    p_η_arr_ = jnp.array(p_η_arr)

    def scan_fn(carry, x):
        y, Z = carry
        ψ, idx = x[:-1], x[-1]
        gof = jnp.linalg.norm(y - Z[jnp.arange(Z.shape[0]), ψ], ord=2) / (2 * jnp.maximum(σ**2, ϵ**2)) - jnp.log(p_η_arr_[idx]) 
        return (y, Z), gof
    xs = jnp.block([C_s, η_idx.reshape(-1, 1)])
    _, gof_list = jax.lax.scan(scan_fn, (y, Z), xs)

    gof_list = np.array(gof_list)
    i_opt = gof_list.argmin()
    η_opt = η_arr[η_idx[i_opt]]

    return η_opt[η_opt != -1]
