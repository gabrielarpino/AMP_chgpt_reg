from matplotlib import pyplot as plt
import pytest
import numpy as np
from scipy.special import comb
# To run pytest, use following command:
# from amp.signal_configuration import configs_arr_for_exact_L, nconfigs_arr_for_exact_L, nconfigs_arr_for_exact_L_old, unif_prior_to_η_ϕ, unif_prior_to_η_ϕ_old
# from amp.posterior import η_to_ψ

# To run main, use following command:
import sys
sys.path.append('/Users/xiaoqiliu/Desktop/3_Changepoints_detection/AMP/amp')
from signal_configuration import unif_prior_to_η_ϕ, configs_arr_for_exact_L, nconfigs_arr_for_exact_L, nconfigs_arr_for_exact_L_old, unif_prior_to_η_ϕ_old
from posterior import η_to_ψ


@pytest.mark.parametrize("L", [2, 3, 5, 7])
@pytest.mark.parametrize("n", [100, 50, 29, 25])
def _test_total_num_configs(L, n):
    num_chgpts = L - 1
    total_num_configs = comb(n-1, num_chgpts)

    num_configs_arr = np.zeros((n, L))
    for i in range(n):
        for l in range(L):
            num_configs_arr[i, l] = comb(i, l) * comb(n-i-1, num_chgpts-l)
    assert np.all(np.sum(num_configs_arr, axis=1) == total_num_configs)
    if L == 3:
        assert np.all(hardcoded_L_3(n) == num_configs_arr)
    if n % 2 == 0:
        assert np.all(num_configs_arr[:n//2, :] == np.fliplr(np.flipud(num_configs_arr[n//2:, :])))
    else:
        assert np.all(num_configs_arr[:n//2, :] == np.fliplr(np.flipud(num_configs_arr[n//2+1:, :])))

@pytest.mark.parametrize("L", [1, 2, 3, 4])
@pytest.mark.parametrize("n", [100, 201])
def test_unif_prior_to_η_ϕ(L, n):
    tot_nconfigs_old, η_old, p_η_old, ϕ_old = unif_prior_to_η_ϕ_old(L, n)
    tot_nconfigs, η, p_η, ϕ = unif_prior_to_η_ϕ(Lmin=1, Lmax=L, Δ=1, n=n, p_l=np.ones((L, ))/L)
    assert np.all(tot_nconfigs_old == tot_nconfigs)
    assert np.allclose(η_old, η)
    assert np.allclose(p_η_old, p_η)
    assert np.allclose(ϕ_old, ϕ)


@pytest.mark.parametrize("L", [1, 2, 3, 4])
@pytest.mark.parametrize("n", [100, 201])
def test_nconfigs_arr_for_exact_L(n, L):
    tot_nconfigs_old, nconfigs_arr_old = nconfigs_arr_for_exact_L_old(n, L)
    η_arr = configs_arr_for_exact_L(n, L, Δ=1)
    nconfigs_arr = nconfigs_arr_for_exact_L(η_arr, n)
    assert η_arr.shape[0] == tot_nconfigs_old
    assert np.all(nconfigs_arr == nconfigs_arr_old)

@pytest.mark.parametrize("L", [1, 2, 3, 4])
@pytest.mark.parametrize("n", [100, 201])
@pytest.mark.parametrize("frac", [0.1, 0.2])
def test_Δ_greater_than_1(n, L, frac):
    Δ = int(n * frac)
    η_arr = configs_arr_for_exact_L(n, L, Δ)
    nconfigs = η_arr.shape[0]
    if nconfigs > 0:
        η_arr_extended = np.concatenate((np.zeros((nconfigs, 1)), 
            η_arr, np.ones((nconfigs, 1))*(n-1)), axis=1)
        assert np.all(np.diff(η_arr_extended, axis=1) >= Δ)

    Lmin = L
    Lmax = L
    tot_nconfigs, η_arr1, p_η_arr, ϕ = unif_prior_to_η_ϕ(Lmin, Lmax, Δ, n)
    assert np.all(η_arr1 == η_arr)

def hardcoded_L_3(n):
    print(f"== L = 3, n = {n} ==")
    L = 3
    num_chgpts = L - 1
    total_num_configs = comb(n-1, num_chgpts)
    print(f"total_num_configs = {total_num_configs}")
    num_configs_arr = np.zeros((n, L))
    num_configs_arr[0, 0] = total_num_configs
    num_configs_arr[-1, -1] = total_num_configs
    num_configs_arr[1, 0] = comb(n-2, 2)
    num_configs_arr[1, 1] = comb(n-2, 1)
    num_configs_arr[-2, -1] = comb(n-2, 2)
    num_configs_arr[-2, -2] = comb(n-2, 1)
    for i in range(2, n-2):
        num_configs_arr[i, 0] = comb(n-i-1, 2)
        num_configs_arr[i, 1] = i * (n-i-1)
        num_configs_arr[i, 2] = comb(i, 2)
    assert np.all(np.sum(num_configs_arr, axis=1) == total_num_configs)
    print(f"num_configs_arr = {num_configs_arr}")
    return num_configs_arr

@pytest.mark.parametrize("L", [2, 3, 4])
@pytest.mark.parametrize("n", [100, 201])
def _test_η_to_ψ(L, n):
    η_arr = unif_prior_to_η_ϕ_old(L, n)[1]
    num_configs = η_arr.shape[0]
    for i in range(num_configs):
        η = η_arr[i, :]
        ψ = η_to_ψ_old(η, n)
        idx_arr = np.arange(1, n)
        assert np.all(η[η != -1] == idx_arr[np.diff(ψ) == 1])
        ψ_new = η_to_ψ(η, n)
        assert np.all(ψ == ψ_new)

def η_to_ψ_old(η, n):
    """
    Convert η (length-(L-1) vector) into Ψ (length-n vector).
    
    η stores the starting index of signal 1 up to signal (L-1).
    -1 in entry j means the config η involves fewer than L signals, 
    excluding signal j.

    ψ stores the signal index underlying each yi.
    """
    # L = η.shape[0] + 1
    num_chgpts = np.sum(η != -1)
    if num_chgpts == 0:
        return np.zeros(n)
    
    ψ = np.ones(n) * num_chgpts
    ψ[:η[0]] = 0
    if num_chgpts > 1:
        for l in range(1, num_chgpts):
            ψ[η[l-1]:η[l]] = l
    assert np.diff(ψ).min() >= 0, "ψ should be non-decreasing"
    return ψ

def test_uniform_p_l():
    L = 3
    n = 300
    Δ = int(n/10)
    tot_nconfigs0, η0, p_η0, ϕ0 = unif_prior_to_η_ϕ(Lmin=1, Lmax=L, Δ=Δ, n=n)
    p_l = np.ones((L, ))/L
    tot_nconfigs1, η1, p_η1, ϕ1 = unif_prior_to_η_ϕ(Lmin=1, Lmax=L, Δ=Δ, n=n, p_l=p_l)
    assert np.all(η0 == η1)
    
    # Confirm η0, η1 should be a concatenation of configs for increasing L:
    num_chgpts_arr0 = np.sum(η0 != -1, axis=1)
    L_arr0 = num_chgpts_arr0 + 1
    idx_arr0 = np.where(np.diff(L_arr0) != 0) 
    idx_arr0 = np.concatenate(([0], idx_arr0[0]+1, [tot_nconfigs0])) # i.e. idx ∈ [num_configs] 
    # of first config that has L signals for L>0

    num_chgpts1 = np.sum(η1 != -1, axis=1)
    L_arr1 = num_chgpts1 + 1
    idx_arr1 = np.where(np.diff(L_arr1) != 0) 
    idx_arr1 = np.concatenate(([0], idx_arr1[0]+1, [tot_nconfigs1]))
    assert np.all(idx_arr0 == idx_arr1)
    sum_p_η_for_l = np.zeros(L)
    for l in range(L):
        p_one_config = 1/L/(idx_arr1[l+1] - idx_arr1[l])
        assert np.allclose(p_η1[idx_arr1[l]: idx_arr1[l+1]], p_one_config)
        sum_p_η_for_l[l] = np.sum(p_η1[idx_arr1[l]: idx_arr1[l+1]])
    assert np.allclose(sum_p_η_for_l, p_l)
    plt.figure()
    plt.plot(p_η0, label="p_l = None")
    plt.plot(p_η1, label="p_l = uniform")
    plt.yscale('log')
    plt.xlabel("config η index")
    plt.ylabel("p_η")
    plt.legend()
    plt.show()

    # ϕ0, ϕ1 shouldnt be the same:
    assert not np.all(ϕ0 == ϕ1)

def test_arbitrary_p_l():
    L = 3
    n = 300
    Δ = int(n/10)
    tot_nconfigs0, η0, p_η0, ϕ0 = unif_prior_to_η_ϕ(Lmin=1, Lmax=L, Δ=Δ, n=n)
    p_l = np.random.random((L, ))
    p_l = p_l/np.sum(p_l)
    tot_nconfigs1, η1, p_η1, ϕ1 = unif_prior_to_η_ϕ(Lmin=1, Lmax=L, Δ=Δ, n=n, p_l=p_l)
    assert np.all(η0 == η1)
    
    # Confirm η0, η1 should be a concatenation of configs for increasing L:
    num_chgpts_arr0 = np.sum(η0 != -1, axis=1)
    L_arr0 = num_chgpts_arr0 + 1
    idx_arr0 = np.where(np.diff(L_arr0) != 0) 
    idx_arr0 = np.concatenate(([0], idx_arr0[0]+1, [tot_nconfigs0])) # i.e. idx ∈ [num_configs] 
    # of first config that has L signals for L>0

    num_chgpts1 = np.sum(η1 != -1, axis=1)
    L_arr1 = num_chgpts1 + 1
    idx_arr1 = np.where(np.diff(L_arr1) != 0) 
    idx_arr1 = np.concatenate(([0], idx_arr1[0]+1, [tot_nconfigs1]))
    assert np.all(idx_arr0 == idx_arr1)
    sum_p_η_for_l = np.zeros(L)
    for l in range(L):
        p_one_config = p_l[l]/(idx_arr1[l+1] - idx_arr1[l])
        assert np.allclose(p_η1[idx_arr1[l]: idx_arr1[l+1]], p_one_config)
        sum_p_η_for_l[l] = np.sum(p_η1[idx_arr1[l]: idx_arr1[l+1]])
    assert np.allclose(sum_p_η_for_l, p_l)
    plt.figure()
    plt.plot(p_η0, label="p_l = None")
    plt.plot(p_η1, label="p_l = uniform")
    plt.yscale('log')
    plt.xlabel("config η index")
    plt.ylabel("p_η")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_arbitrary_p_l()
    test_Δ_greater_than_1(20, 3, 0.5)
    unif_prior_to_η_ϕ(Lmin=1, Lmax=3, Δ=1, n=20, p_l=np.ones((3, ))/3)
    hardcoded_L_3(10)
    test_total_num_configs(4, 9)