import jax
import jax.numpy as jnp
import numpy.random as nprandom
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
from amp import ϵ_0
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from amp.changepoint_jax import *
from amp.fully_separable import psd_mat
import random
import matplotlib.pyplot as plt
import amp.covariances
import unittest

class TestWoodbury(unittest.TestCase):
    def test_woodbury(self):
        n = 200
        L = 2
        σ = 0.5
        ρ = 2.0 * jnp.eye(L)
        ρ = ρ.at[0, 0].set(1.0)
        ρ = ρ.at[0, 1].set(1.0)
        ρ = ρ.at[1, 0].set(1.0)
        ν = psd_mat(L)
        κ_T = psd_mat(L)
        nprandom.seed(10)

        indx = jnp.arange(0,n).reshape((n, ))
        V = nprandom.normal(size=(1, L))
        u = nprandom.normal(size=(n, 1))
        C_full = jnp.triu(jnp.ones((n, n)), k=0).astype(int)

        j_crit = int(2*n/3)
        C_s = C_full[jnp.array([int(n/3), int(2*n/3)])]
        C_0 = C_s[1]

        Cov_V_Y_ = amp.covariances.Cov_V_Y(j_crit, C_0, n, ρ, ν)
        Cov_Z_Y_ = amp.covariances.Cov_Z_Y(j_crit, C_0, n, ρ)

        Σ_Y = amp.covariances.Σ_Y(C_0, n, ρ, σ)
        B = jnp.block([
            [ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T, Cov_V_Y_],
            [Cov_V_Y_.T, Σ_Y]
        ])

        A = jnp.block([
            [ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T, jnp.zeros((L, n))],
            [jnp.zeros((L, n)).T, Σ_Y]
        ])

        A_inv = jnp.block([
            [jnp.linalg.inv(ν.T @ jnp.linalg.inv(ρ) @ ν + κ_T), jnp.zeros((L, n))],
            [jnp.zeros((L, n)).T, amp.covariances.Σ_Y_inv(C_0, n, ρ, σ)]
        ])

        self.assertTrue(jnp.allclose(A_inv, jnp.linalg.inv(A)))
        res = amp.covariances.Σ_V_Y_inv(j_crit, C_0, n, ρ, σ, ν, κ_T)
        self.assertTrue(jnp.max(res - jnp.linalg.inv(B)) < 5e-5) # Inverse directly through woodbury is more accurate.
