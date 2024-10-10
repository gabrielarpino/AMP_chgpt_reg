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
import amp.posterior
import unittest


class TestPosterior(unittest.TestCase):
    def setUp(self):
        self.n = 20
        self.L = 2
        self.σ = 0.1
        self.ρ = 1.0 * jnp.eye(self.L)
        self.ρ = self.ρ.at[0, 0].set(0.5)
        self.ρ = self.ρ.at[0, 1].set(-0.2)
        self.ρ = self.ρ.at[1, 0].set(-0.2)
        nprandom.seed(10)
        self.ν = psd_mat(self.L)
        self.κ_T = psd_mat(self.L)
        self.j = 4

    def test_posterior(self):
        nprandom.seed(10)
        V = nprandom.normal(size=(self.n, self.L))
        u = nprandom.normal(size=(self.n, 1))
        Y = nprandom.normal(size=(self.n, 1))
        C_full = jnp.triu(jnp.ones((self.n, self.n)), k=0).astype(int)  # Here we are creating the matrix of all possible C's. This assumes that one changepoint happens forsure between 0≤t≤n-1. 
        C_s_1 = C_full[jnp.array([int(self.n/5), int(self.n/3), int(2*self.n/3)])] # Ask the AMP to search over 3 possible changepoint locations
        C_s_2 = C_full[jnp.array([int(self.n/10), int(self.n/5), int(4*self.n/5)])] # Ask the AMP to search over 3 possible changepoint locations
        post_1 = amp.posterior.full_posterior(C_s_1, V, Y, self.n, self.ρ, self.σ, self.ν, self.κ_T)
        post_2 = amp.posterior.full_posterior(C_s_2, V, Y, self.n, self.ρ, self.σ, self.ν, self.κ_T)
        self.assertTrue(post_1.shape == (3, ))
        self.assertTrue(jnp.allclose(jnp.sum(post_1, axis=0), 1))
        self.assertTrue(not jnp.allclose(post_1, post_2))

    def test_fast_posterior(self):
        nprandom.seed(10)
        self.n = 50
        V = nprandom.normal(size=(self.n, self.L))
        u = nprandom.normal(size=(self.n, 1))
        Y = nprandom.normal(size=(self.n, 1))
        C_full = jnp.triu(jnp.ones((self.n, self.n)), k=0).astype(int)  # Here we are creating the matrix of all possible C's. This assumes that one changepoint happens forsure between 0≤t≤n-1. 
        C_s_1 = C_full[int(3*self.n/8):int(5*self.n/8)]
        C_s_1_comparable = C_full[jnp.array([int(self.n/5), int(self.n/3), int(2*self.n/3)])] # Ask the AMP to search over 3 possible changepoint locations
        C_s_2 = C_full[jnp.array([int(self.n/10), int(self.n/5), int(4*self.n/5)])] # Ask the AMP to search over 3 possible changepoint locations
        post_1_full = amp.posterior.full_posterior(C_s_1, V, Y, self.n, self.ρ, self.σ, self.ν, self.κ_T)
        post_1_fast = amp.posterior.fast_posterior(C_s_1, V, Y, self.n, self.ρ, self.σ, self.ν, self.κ_T)
        post_2_full = amp.posterior.full_posterior(C_s_2, V, Y, self.n, self.ρ, self.σ, self.ν, self.κ_T)
        post_2_fast = amp.posterior.fast_posterior(C_s_2, V, Y, self.n, self.ρ, self.σ, self.ν, self.κ_T)
        post_1_fast_comparable = amp.posterior.fast_posterior(C_s_1_comparable, V, Y, self.n, self.ρ, self.σ, self.ν, self.κ_T)
        self.assertTrue(post_1_fast_comparable.shape == (3, ))
        self.assertTrue(jnp.allclose(jnp.sum(post_1_fast, axis=0), 1))
        self.assertTrue(not jnp.allclose(post_1_fast_comparable, post_2_fast))
        self.assertTrue(jnp.allclose(post_1_full, post_1_fast, atol = 5e-3))
        self.assertTrue(jnp.allclose(post_2_full, post_2_fast, atol = 5e-3))