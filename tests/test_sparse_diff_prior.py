import sys
import numpy as np
import pytest
sys.path.append('/Users/xiaoqiliu/Desktop/3_Changepoints_detection/AMP/')
from amp.signal_priors import SparseDiffSignal, SparseDiffSignal_old

@pytest.mark.parametrize("var_1", [1, 0.9, 2])
@pytest.mark.parametrize("σ_w", [1, 0.9, 3])
@pytest.mark.parametrize("α", [0.1, 0.5, 0.9])
def test_cov(var_1, σ_w, α):
    """Check cov matches hard coded simple cases."""
    def test_individual_cov(η, cov, is_change):
        assert np.allclose(cov, cov.T), "cov should be symmetric"
        assert np.all(np.diag(cov) == var_1), "cov diagonal should be var_1"
        assert np.all(cov > 0), "cov should be positive definite"
        num_changes = np.sum(is_change)
        assert np.allclose(np.unique(cov), var_1 * η ** np.arange(num_changes, -1, -1)), \
            "cov should only contain powers of η and 1=η^0"
    ############################## L = 2 #####################################
    print('Checking L=2')
    # is_change_seq = array([[0],
    #    [1]])
    δ = 1
    ρ_1 = var_1 / δ
    signal = SparseDiffSignal_old(δ, ρ_1, σ_w, α)
    signal_v1 = SparseDiffSignal(var_1, σ_w, α, L=2)
    expected_cov = signal.cov
    cov = signal_v1.cov
    assert np.allclose(cov, expected_cov)
   
    expected_cov_seq = np.concatenate(
        (signal.ρ_B_same[np.newaxis, :, :], signal.ρ_B_diff[np.newaxis, :, :]), axis=0)
    cov_seq = signal_v1.cov_seq
    for i_cov in range(cov_seq.shape[0]):
        test_individual_cov(signal_v1.η, cov_seq[i_cov], signal_v1.is_change_seq[i_cov])
    assert np.allclose(cov_seq, expected_cov_seq)

    ############################## L = 3 #####################################
    print('Checking L=3')
    # is_change_seq = array([[0, 0],
    #    [0, 1],
    #    [1, 0],
    #    [1, 1]])
    L = 3
    signal_v1 = SparseDiffSignal(var_1, σ_w, α, L)
    η = signal_v1.η
    # Hard code covariances:
    cov_00 = var_1 * np.ones((L, L))
    cov_01 = var_1 * np.array([[1, 1, η],
                               [1, 1, η],
                               [η, η, 1]])
    cov_10 = var_1 * np.array([[1, η, η],
                               [η, 1, 1],
                               [η, 1, 1]])
    cov_11 = var_1 * np.array([[1, η, η**2],
                               [η, 1, η],
                               [η**2, η, 1]])
    expected_cov_seq = np.concatenate((cov_00[np.newaxis, :, :],
                                       cov_01[np.newaxis, :, :],
                                       cov_10[np.newaxis, :, :],
                                       cov_11[np.newaxis, :, :]), axis=0)
    cov_seq = signal_v1.cov_seq
    assert np.allclose(cov_seq, expected_cov_seq)
    for i_cov in range(cov_seq.shape[0]):
        test_individual_cov(η, cov_seq[i_cov], signal_v1.is_change_seq[i_cov])

    ############################## L = 4 #####################################
    print('Checking L=4')
    # is_change_seq = array([[0, 0, 0],
    #    [0, 0, 1],
    #    [0, 1, 0],
    #    [0, 1, 1],
    #    [1, 0, 0],
    #    [1, 0, 1],
    #    [1, 1, 0],
    #    [1, 1, 1]])
    L = 4
    signal_v1 = SparseDiffSignal(var_1, σ_w, α, L)
    η = signal_v1.η
    # Hard code covariances:
    cov_000 = var_1 * np.ones((L, L))
    cov_001 = var_1 * np.array([[1, 1, 1, η],
                                [1, 1, 1, η],
                                [1, 1, 1, η],
                                [η, η, η, 1]])
    cov_010 = var_1 * np.array([[1, 1, η, η],
                                [1, 1, η, η],
                                [η, η, 1, 1],
                                [η, η, 1, 1]])
    cov_011 = var_1 * np.array([[1, 1, η, η**2],
                                [1, 1, η, η**2],
                                [η, η, 1, η],
                                [η**2, η**2, η, 1]])
    cov_100 = var_1 * np.array([[1, η, η, η],
                                [η, 1, 1, 1],
                                [η, 1, 1, 1],
                                [η, 1, 1, 1]])
    cov_111 = var_1 * np.array([[1, η, η**2, η**3],
                                [η, 1, η, η**2],
                                [η**2, η, 1, η],
                                [η**3, η**2, η, 1]])
    cov_110 = var_1 * np.array([[1, η, η**2, η**2],
                                [η, 1, η, η],
                                [η**2, η, 1, 1],
                                [η**2, η, 1, 1]])
    cov_101 = var_1 * np.array([[1, η, η, η**2],
                                [η, 1, 1, η],
                                [η, 1, 1, η],
                                [η**2, η, η, 1]])
    expected_cov_seq = np.concatenate((cov_000[np.newaxis, :, :],
                                       cov_001[np.newaxis, :, :],
                                       cov_010[np.newaxis, :, :],
                                       cov_011[np.newaxis, :, :],
                                       cov_100[np.newaxis, :, :],
                                       cov_101[np.newaxis, :, :],
                                       cov_110[np.newaxis, :, :],
                                       cov_111[np.newaxis, :, :]), axis=0)
    cov_seq = signal_v1.cov_seq
    assert np.allclose(cov_seq, expected_cov_seq)
    for i_cov in range(cov_seq.shape[0]):
        test_individual_cov(η, cov_seq[i_cov], signal_v1.is_change_seq[i_cov])

@pytest.mark.parametrize("var_1", [1, 0.9, 2])
@pytest.mark.parametrize("σ_w", [1, 0.9, 3])
@pytest.mark.parametrize("α", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("L", [2, 3, 4])
def test_sample(var_1, σ_w, α, L):
    num_samples = 10000
    B = SparseDiffSignal(var_1, σ_w, α, L).sample(num_samples)
    assert B.shape == (num_samples, L)
    for l in range(1, L):
        is_different = B[:, l] != B[:, l-1]
        assert np.abs(np.mean(is_different) - α) < 3 * 1e-2, \
            "Fraction of different entries should be α"
        assert np.abs(np.mean(B[is_different, l])) < 5 * 1e-2, \
            "Differing entries should be zero mean"
        assert np.abs(np.mean(B[is_different, l] ** 2) - var_1) < 8 * 1e-2, \
            "Differing entries are normalised to have variance var_1" 

    # Confirm that 1 sample doesnt cause problems:
    B = SparseDiffSignal(var_1, σ_w, α, L).sample(1)

if __name__ == "__main__":
    print("Running test...")
    test_cov(1, 2, 0.5)
    print("passed!")