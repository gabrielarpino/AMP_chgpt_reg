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

class TestCovariances(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.L = 2
        self.σ = 0.1
        self.ρ = 1.0 * jnp.eye(self.L)
        self.ρ = self.ρ.at[0, 0].set(0.5)
        self.ρ = self.ρ.at[0, 1].set(-0.2)
        self.ρ = self.ρ.at[1, 0].set(-0.2)
        self.ν = psd_mat(self.L)
        self.κ_T = psd_mat(self.L)
        self.j = 4
        self.indx = jnp.arange(0,self.n).reshape((self.n, ))
        self.V = nprandom.normal(size=(1, self.L))
        self.u = nprandom.normal(size=(self.n, 1))
        self.C_full = jnp.triu(jnp.ones((self.n, self.n)), k=0).astype(int)

    def test_Cov_V_Y(self):
        C_s = self.C_full[jnp.array([int(self.n/3), int(2*self.n/3)])]
        C_0 = C_s[0]
        C_1 = C_s[1]
        Σ_0 = jax.vmap(amp.covariances.Cov_V_Y, (0, None, None, None, None), 0)(self.indx, C_0, self.n, self.ρ, self.ν).reshape((self.n * self.L, self.n))
        Σ_1 = jax.vmap(amp.covariances.Cov_V_Y, (0, None, None, None, None), 0)(self.indx, C_1, self.n, self.ρ, self.ν).reshape((self.n * self.L, self.n))
        self.assertFalse(jnp.allclose(Σ_0, Σ_1)) # They should differ at indices j.
    
    def test_Σ_V_Y(self):
        C_s = self.C_full[jnp.array([int(self.n/3), int(2*self.n/3)])]
        C_0 = C_s[0]
        Σ_V_Y_full = amp.covariances.Σ_V_Y_full(C_0, self.n, self.ρ, self.σ, self.ν, self.κ_T)
        self.assertEqual(Σ_V_Y_full.shape, (self.n*(self.L+1), self.n*(self.L+1)))
        self.assertTrue(jnp.allclose(Σ_V_Y_full, Σ_V_Y_full.T))

    def test_Σ_Y(self):
        C_s = self.C_full[jnp.array([int(self.n/3), int(2*self.n/3)])]
        C_0 = C_s[0]
        Σ_Y = amp.covariances.Σ_Y(C_0, self.n, self.ρ, self.σ)
        self.assertTrue(Σ_Y.shape == (self.n, self.n))
        self.assertTrue(jnp.allclose(Σ_Y[:int(self.n/3), :int(self.n/3)], jnp.eye(int(self.n/3)) * (self.ρ[0, 0] + self.σ**2)))



