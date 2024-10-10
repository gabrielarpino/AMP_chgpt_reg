# Only works on numpy before version 1.9, because they changed the np.apply_along_axis function. 

import numpy as np
import numpy.random as nprandom
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import scipy
from tqdm import tqdm
from amp import ϵ_0
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from amp.changepoint import *
from amp.fully_separable import psd_mat
import random
import matplotlib.pyplot as plt

n = 8
L = 2
σ = 0.0
ρ = 1.0 * np.eye(L)
ρ[0, 1] = -0.2
ρ[1, 0] = -0.2
ν = psd_mat(L)
κ_T = psd_mat(L)
j = 3

# Form C: 
# ξ = random.randint(1, n-1) # TODO: have to test things when ξ is at the endpoints
ξ = 6
C = np.zeros((n, 1)).astype(int)
C[ξ:] = np.ones((n-ξ, 1)).astype(int)

# Form C_2: 
C_2 = np.zeros((n, 1)).astype(int)
C_2[ξ + 1:] = np.ones((n-ξ-1, 1)).astype(int)

assert not np.all(C == C_2)
  
# Test the form_Σ function
Σ = form_Σ(C, ξ, L, σ, ρ, ν, κ_T)
Σ_2 = form_Σ(C_2, ξ, L, σ, ρ, ν, κ_T) 
assert np.all(np.linalg.eigvals(Σ + ϵ_0 * np.eye(Σ.shape[0])) >= 0)
assert not np.allclose(Σ, Σ_2) # We expect both Σ to differ at indices j where their corresponding changepoint vector C differs.  

# Test the form_λ function
V = nprandom.normal(size=(1, L))
u = nprandom.normal(size=(n, 1))
Σ = form_Σ(C, ξ, L, σ, ρ, ν, κ_T)
Σ_b = form_Σ(C, ξ-1, L, σ, ρ, ν, κ_T) 
λ = form_λ(Σ, V, u, L)
λ_b = form_λ(Σ_b, V, u, L)
assert λ.shape[0] == L
assert Σ.shape == (2*L + n, 2*L + n)
assert not np.allclose(Σ, Σ_b) # We expect them to differ at index ξ-1 vs ξ, because that is where the changepoint occurs
assert not np.allclose(λ, λ_b)

# Test the log_N fun
assert not np.allclose(log_N(Σ, V, u, L), log_N(Σ_b, V, u, L))

# Test vectorized versions of the above
C_s = np.triu(np.ones((n, n)), k=0).astype(int)  # Here we are creating the matrix of all possible C's. This assumes that one changepoint happens forsure between 0≤t≤n-1. 
Σ_s = np.apply_along_axis(form_Σ, 0, C_s, j, L, σ, ρ, ν, κ_T)
assert Σ_s.shape == (2*L + n, 2*L + n, 8)
assert not np.allclose(Σ_s[:, :, j-1], Σ_s[:, :, j])
assert np.allclose(Σ_s[:, :, j], Σ_s[:, :, j+1])

# CONTINUE HERE
# # This does not work because apply along axis expects that the function accepts 1-D arrays! Might have to flatten and unflatten? 
# λ_s = np.apply_along_axis(form_λ, 0, Σ_s, V, u, L)

# Test the E_Z function function
C_s = np.triu(np.ones((n, n)), k=0).astype(int)  # Here we are creating the matrix of all possible C's. This assumes that one changepoint happens forsure between 0≤t≤n-1. 
Σ_s, λ_s = np.apply_along_axis(form_Σ_submatrix_λ, 0, C_s, j, V, u, L, σ, ρ, ν, κ_T) # Raises a deprecated warning. 
# We expect that Σ_s[j-1] is different from Σ_s[j], and Σ_s[k] stays the same for k ≥ j. 
assert not np.allclose(Σ_s[j-1], Σ_s[j])
assert np.allclose(Σ_s[j], Σ_s[j+1])
assert not np.allclose(λ_s[j-1], λ_s[j])
assert np.allclose(λ_s[j], λ_s[j+1])

# Test the long vectorized form function in E_Z
Σ_s, λ_s, log_N_s = np.apply_along_axis(form_Σ_submatrix_λ_log_N, 0, C_s, j, V, u, L, σ, ρ, ν, κ_T) # apply to the columns of C_s. 
assert log_N_s.shape == (n, )
assert log_N_s.shape == λ_s.shape

E_Z_res = E_Z(V, j, u, L, σ, ρ, ν, κ_T)
assert E_Z_res.shape == (1, L)

# test indivudual g. 
g_j = define_g_chgpt(L, σ, ρ, ν, κ_T)
assert g_j(V, j, u).shape == (1, L)

# Test full g with the infices. 
Θ = nprandom.normal(size=(n, L))
Y = nprandom.normal(size=(n, 1))
# R̂ = g_chgpt_full(Θ, Y, L, σ, ρ, ν, κ_T)
g_ = define_g_chgpt(L, σ, ρ, ν, κ_T)
indx = np.arange(0, Θ.shape[0]).reshape((Θ.shape[0], 1))
g_wrapper = lambda x: g_(x[0:L].reshape((1, L)), int(x[L]), Y).reshape(L, )
g_full_res = np.apply_along_axis(g_wrapper, 1, np.concatenate((Θ, indx), axis=1))
assert g_full_res.shape == (n, L)
assert g_chgpt_full(Θ, Y, L, σ, ρ, ν, κ_T).shape == (n, L)

# Test g_j_chgpt_full
res = g_j_chgpt_full(j, Θ, Y, L, σ, ρ, ν, κ_T)
assert res.shape == (n, L)

# Test the inner workings of the AMP iteration non-sep
Ω = ν.T @ np.linalg.inv(ρ) @ ν + κ_T
inner = lambda i: 1/n * Θ.T @ g_j_chgpt_full(i, Θ, Y, L, σ, ρ, ν, κ_T)
# indx = np.arange(0, Θ.shape[0]).reshape((Θ.shape[0], ))
# sum_list = np.apply_along_axis(inner, 0, indx)
C =  1/n * sum([inner(i) for i in range(0, n)]).T @ np.linalg.pinv(Ω).T # Should be tested. 