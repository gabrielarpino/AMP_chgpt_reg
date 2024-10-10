import jax
import jax.numpy as np
import numpy.random as nprandom
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import scipy
from tqdm import tqdm
from amp import ϵ_0
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from amp.changepoint_jax import *
from amp.fully_separable import psd_mat
import random
import matplotlib.pyplot as plt

n = 10
L = 2
σ = 0.0
ρ = 1.0 * np.eye(L)
ρ = ρ.at[0, 1].set(-0.2) # Now compatible with jax
ρ = ρ.at[1, 0].set(-0.2)
ν = psd_mat(L)
κ_T = psd_mat(L)
j = 3

# Form C: 
ξ = random.randint(1, n-1) # TODO: have to test things when ξ is at the endpoints
C = np.zeros((n, 1)).astype(int)
C = C.at[ξ:].set(np.ones((n-ξ, 1)).astype(int))

# Form C_2: 
C_2 = np.zeros((n, 1)).astype(int)
C_2 = C_2.at[ξ + 1:].set(np.ones((n-ξ-1, 1)).astype(int))

assert not np.all(C == C_2)
  
# Test the form_Σ function
Σ = form_Σ(C, ξ, L, σ, ρ, ν, κ_T)
Σ_2 = form_Σ(C_2, ξ, L, σ, ρ, ν, κ_T) 
# assert np.all(np.linalg.eigvals(Σ + ϵ_0 * np.eye(Σ.shape[0])) >= 0) # Why does this fail sometimes? 
assert not np.allclose(Σ, Σ_2) # We expect both Σ to differ at indices j where their corresponding changepoint vector C differs.  

# Test the form_Σ_submatrix_λ function
V = nprandom.normal(size=(1, L))
u = nprandom.normal(size=(n, 1))
Σ, λ, info = form_Σ_submatrix_λ(C, ξ, V, u, L, σ, ρ, ν, κ_T) 
Σ_b, λ_b, info = form_Σ_submatrix_λ(C, ξ-1, V, u, L, σ, ρ, ν, κ_T) 
assert λ.shape[0] == L
assert Σ.shape == (L + n, L + n)
assert not np.allclose(Σ, Σ_b)
assert not np.allclose(λ, λ_b)

# Test the inner workings of E_Z. Just some playing around scripts. 
C_s = np.triu(np.ones((n, n)), k=0).astype(int)  # Here we are creating the matrix of all possible C's. This assumes that one changepoint happens forsure between 0≤t≤n-1. 
#### RES DIFFERS DEPENDING ON WHETHER 0 here or 1 ! in in_axes! The axis index should be zero tho, start with chgpt at 0, so all ones vector. 
λ_log_N_s = jax.vmap(form_λ_log_N, (0, None, None, None, None, None, None, None, None), 0)(C_s, j, V, u, L, σ, ρ, ν, κ_T)
assert λ_log_N_s.shape == (n, L + 1)
λ_s = λ_log_N_s[:, :L]
log_N_s = λ_log_N_s[:, L:]
log_num_arr, sign_arr = logsumexp(a = log_N_s, axis=0, b = λ_s, return_sign=True)
log_denom = logsumexp(a = log_N_s, axis=0, return_sign=False)
res = np.multiply(sign_arr, np.exp(log_num_arr - log_denom))

# Test E_Z explicitly
res = E_Z(V, j, u, L, σ, ρ, ν, κ_T)
assert res.shape == (1, L)

# Test g_chgpt
res = g_chgpt(V, j, u, L, σ, ρ, ν, κ_T)
assert res.shape == (1, L)

# Test g_chgpt_tensorized
# Θ = nprandom.normal(size=(n, L))
# Y = nprandom.normal(size=(n, 1))
# indx = np.arange(0, Θ.shape[0]).reshape((Θ.shape[0], 1))
# g_chgpt_tensorized = jax.jit(g_chgpt_full())
# res = g_chgpt_tensorized(Θ, indx, Y, L, σ, ρ, ν, κ_T).reshape((n, L)) # Runs well even on n=200

# # Test the jacobian of g. This is relatively fast, but could be done faster? 
# dgdθ = jax.jacfwd(g_chgpt, 0)
# jacobian = jax.jit(lambda j: dgdθ(V, j, u, L, σ, ρ, ν, κ_T).reshape((L, L)))
# C = 1/n * sum([jacobian(j) for j in range(0, n)])

# ### --- C by automatic differentiation, returns NANs in the jupyter notebook for now --- ###
# # For defining all at once first, the first 5 lines take some time. but then, calling C_fun_jit is very fast after the first time. Should run it once with random params to compile.  
# # Not sure how much I trust this, because it requires that the conjugate gradient converges. 
# # ν = np.zeros((2, 2))
# # κ_T = ρ
# dgdθ = jax.jacfwd(g_chgpt, 0)
# jacobian = lambda V, j, u, ν, κ_T: dgdθ(V, j, u, L, σ, ρ, ν, κ_T).reshape((L, L))
# def C_fun(Θ_, Y_, ν, κ_T):
#     indx = np.arange(0, Θ_.shape[0]).reshape((Θ_.shape[0], 1))
#     jac_map = jax.vmap(jacobian, (0, 0, None, None, None), 0)
#     return 1/n * np.sum(jac_map(Θ_, indx, Y_, ν, κ_T), axis=0)
# C_tensorized = jax.jit(C_fun)
# C_AD = C_tensorized(Θ, Y, ν, κ_T) # 1 minute on n = 200

# ### --- C by approximate stein's lemma --- ###
# g_j_chgpt_tensorized = jax.jit(g_j_chgpt_full())
# Ω = ν.T @ np.linalg.inv(ρ) @ ν + κ_T
# inner = lambda i: 1/n * Θ.T @ g_j_chgpt_tensorized(Θ, i, Y, L, σ, ρ, ν, κ_T).reshape((n, 2))
# inner_map = jax.vmap(inner, 0, 0)
# indx = np.arange(0, Θ.shape[0]).reshape((Θ.shape[0], ))
# # # sum_list = np.apply_along_axis(inner, 0, indx)
# C_steins =  1/n * np.sum(inner_map(indx), axis=0).T @ np.linalg.pinv(Ω).T # THIS IS THE SPEED BOTTLENECK. Also double check if theory correct. 
