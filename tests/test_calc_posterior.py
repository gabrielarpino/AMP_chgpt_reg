import sys
sys.path.append('/Users/xiaoqiliu/Desktop/3_Changepoints_detection/AMP/')
from amp import posterior, signal_configuration
import numpy as np
import pytest
import jax.numpy as jnp

def test_basics():
    L = 2
    n = 100
    σ = 1
    # Changepoint locations
    Δ = lambda n: int(n/5)
    print("Generating signal config...")
    C_s_function = lambda n: signal_configuration.generate_C_stagger(n, L, Δ(n))
    C_s = C_s_function(n)
    # Δ = lambda n: int(n/5)
    # C_s_function = lambda n: amp.signal_configuration.generate_C_distanced(n, L, Δ = Δ(n))
    # C_s = C_s_function(n)
    # plt.imshow(C_s, cmap="gray")
    # plt.show()

    true_chgpt_locations = lambda n: [int(n/3)]
    # Generate C_true from the true_chgpt_locations function
    C_true = np.zeros((n, )).astype(int)
    loc = 0
    for j_ in range(len(true_chgpt_locations(n)) + 1):
        if len(true_chgpt_locations(n)) == 0:
            break
        if j_ >= len(true_chgpt_locations(n)):
            C_true[loc:] = j_
            break
        C_true[loc:true_chgpt_locations(n)[j_]] = j_
        loc = true_chgpt_locations(n)[j_]

    ϕ = signal_configuration.C_to_marginal(C_s) 
    ρ = jnp.eye((L)) # np.eye causes jax Tracer error
    ν = np.random.randn(L, L)
    ν = ν @ ν.T / 2 # make ν positive definite
    κ_T = np.random.randn(L, L)
    κ_T = κ_T @ κ_T.T / 2 # make κ_T positive definite
    Y = np.random.randn(n, 1)
    V_θ = np.random.randn(n, L)
    posterior.compute_posterior(C_s, V_θ, Y, n, ρ, σ, ν, κ_T)

if __name__ == "__main__":
    test_basics()