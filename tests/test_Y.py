import numpy.random as nprandom
import numpy as np
from amp import q, MSE, norm_sq_corr, run_chgpt_GAMP_jax, run_GAMP, PAL, signal_configuration
import jax
import unittest
import jax.numpy as jnp


class TestY(unittest.TestCase):
    def setUp(self):
        pass

    def test_Y(self):
        nprandom.seed(1)
        p = 200
        σ = 0.0 # noise standard deviation
        L = 2 # num signals. L-1 changepoints
        ϕ = 1 - 1/4
        δ = 0.75

        B̃_cov = np.eye(L)
        B̃ = nprandom.multivariate_normal(np.zeros(L), B̃_cov, size=p)
        B̂_0 = nprandom.multivariate_normal(np.zeros(L), B̃_cov, size=p)

        n = int(δ * p)
        ρ = 1/δ * B̃_cov # 1/δ * Covariance matrix of each row of B_0, independent for now but should be generalized later
        Y = np.zeros((n, 1))
        B_bar_mean = np.array([0, 0])
        # η = nprandom.normal(0.0, σ, (n, 1)) # noise
        X = nprandom.normal(0, np.sqrt(1/n), (n, p))

        # Select where the true changepoint will be, as that will highly affect the result. 
        C_full = np.triu(np.ones((n, n)), k=0).astype(int) # Here we are creating the matrix of all possible C's. This assumes that one changepoint happens forsure between 0≤t≤n-1. 
        C_s = C_full[np.array([int(n/4)])] # Ask the AMP to search over 1 possible changepoint locations
        # C_true = C_s[0] # Select the true change point to lie in the middle

        # Generate the observation vector Y
        Θ = X @ B̃
        self.assertTrue(Θ.shape == (n, L))
        Y = q(Θ, C_s, σ).sample()
        self.assertTrue(np.any(Y - Θ[:, 0].reshape((n, 1)))) # Check that this is not all zeros
        self.assertTrue(np.any(Y - Θ[:, 1].reshape((n, 1)))) # Check that this is not all zeros
        self.assertTrue(np.all(Y[:int(n/4)] == Θ[:, 0].reshape((n, 1))[:int(n/4)]))
        self.assertTrue(np.all(Y[int(n/4):] == Θ[:, 1].reshape((n, 1))[int(n/4):]))
        Y_2 = q(Θ, C_s[0], σ).sample() 
        self.assertTrue(np.any(Y_2 - Θ[:, 0].reshape((n, 1)))) # Check that this is not all zeros
        self.assertTrue(np.any(Y_2 - Θ[:, 1].reshape((n, 1)))) # Check that this is not all zeros
        self.assertTrue(np.all(Y_2[:int(n/4)] == Θ[:, 0].reshape((n, 1))[:int(n/4)]))
        self.assertTrue(np.all(Y_2[int(n/4):] == Θ[:, 1].reshape((n, 1))[int(n/4):]))
        self.assertTrue(np.allclose(Y, Y_2))

        # Test with noise
        σ = 0.01 # noise standard deviation
        Y = q(Θ, C_s, σ).sample()
        self.assertTrue(Y.shape == (n, 1)) # Check that this is not all zeros
        self.assertTrue(np.all(np.abs(Y - Θ[:, 0].reshape((n, 1))) > 0)) # Check that this is not all zeros
        self.assertTrue(np.all(np.abs(Y - Θ[:, 1].reshape((n, 1))) > 0)) # Check that this is not all zeros

    def test_Y_L_3(self):
        p = 200
        σ = 0.0 # noise standard deviation
        L = 3 # num signals. L-1 changepoints
        ϕ = 1 - 1/4
        δ = 0.75

        B̃_cov = np.eye(L)
        B̃ = nprandom.multivariate_normal(np.zeros(L), B̃_cov, size=p)
        B̂_0 = nprandom.multivariate_normal(np.zeros(L), B̃_cov, size=p)

        n = int(δ * p)
        ρ = 1/δ * B̃_cov # 1/δ * Covariance matrix of each row of B_0, independent for now but should be generalized later
        Y = np.zeros((n, 1))
        B_bar_mean = np.array([0, 0])
        η = nprandom.normal(0.0, σ, (n, 1)) # noise
        X = nprandom.normal(0, np.sqrt(1/n), (n, p))

        # Select where the true changepoint will be, as that will highly affect the result. 
        C_full = np.triu(np.ones((n, n)), k=0).astype(int) # Here we are creating the matrix of all possible C's. This assumes that one changepoint happens forsure between 0≤t≤n-1. 
        C_s = C_full[np.array([int(n/4)])] # Ask the AMP to search over 1 possible changepoint locations
        # C_true = C_s[0] # Select the true change point to lie in the middle

        # Generate the observation vector Y
        Θ = X @ B̃
        η = nprandom.normal(loc = 0, scale = σ, size=(n, 1))
        η = jnp.array(η)
        # Y = (self.Θ[jnp.arange(self.n), self.C_s] + self.η.T).T
        # Y = jax.numpy.take_along_axis(self.Θ, self.C_s, axis=1)[:, 0].reshape((self.n, 1)) + self.η # Does not produce jax trace errors. 
        # HACK FOR L = 2: 
        # Y = (jax.numpy.multiply(Θ[:, 0],jax.numpy.ones((n,)) - C_s.reshape((n,))) + jax.numpy.multiply(Θ[:, 1], C_s.reshape((n,))) + η.flatten()).reshape((n, 1)) # This hack works for L = 2, have to generalize for L > 2. 

        τ = 2
        # C_full_2 = generate_C_s.generate_C_full(n, L)
        C_stagger = signal_configuration.generate_C_stagger(n, L, τ) # Tested

        Y_2 = q(Θ, C_stagger[0], σ).sample() 

        self.assertTrue(Y_2.shape == (n, 1)) # Check that this is not all zeros
        self.assertTrue(np.any(Y_2 - Θ[:, 0].reshape((n, 1)))) # Check that this is not all zeros
        self.assertTrue(np.any(Y_2 - Θ[:, 1].reshape((n, 1)))) # Check that this is not all zeros
        self.assertTrue(np.any(Y_2 - Θ[:, 2].reshape((n, 1)))) # Check that this is not all zeros
        self.assertTrue(np.all(Y_2[:τ] == Θ[:, 0].reshape((n, 1))[:τ]))
        self.assertTrue(np.all(Y_2[τ:2 * τ] == Θ[:, 1].reshape((n, 1))[τ:2 * τ]))
        self.assertTrue(np.all(Y_2[2 * τ:] == Θ[:, 2].reshape((n, 1))[2 * τ:]))
        return


# This definition of Y has to be generalized for L > 2! And still work with jax. Use something like np.take. 
