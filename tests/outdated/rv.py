from amp import *
a = Gaussian(np.zeros((1, 2)), np.eye(2))
W = np.random.rand(2, 2)
b = np.random.rand(1, 2)
z = Gaussian(np.random.rand(1, 2), np.eye(2))
c = a @ W
d = c + b
y = d + z
y = d + z
y_2 = y + y

Z = Gaussian(np.zeros((1, 2)), np.eye(2))
ρ = np.random.rand(2, 2)
ν_Θ = np.random.rand(2, 2)
G_Θ = Gaussian(np.zeros((1, 2)), 100*np.eye(2))
V_Θ = Z @ ρ @ ν_Θ + G_Θ

B = Gaussian(np.zeros((1, 2)), 0.1*np.eye(2))
ν_B = np.random.rand(2, 2)
G_B = Gaussian(np.zeros((1, 2)), 10 * np.eye(2))
V_B = B @ ν_B + G_B

print(B.sample())
print(V_B.sample(size=10))

### Test for containment
V2 = B @ ρ + G_B
V3 = Z + G_Θ
assert B in V_B
assert V2 in V_B
assert V_B in V2
assert V_B not in V3
assert V3 not in V_B

### Gaussian pdf evaluation
print(Z.pdf(np.zeros((1, 2))))
assert Z.pdf(np.zeros((1, 2))) == Z.pdf(np.zeros(2))
print(V_B.pdf(np.zeros((1, 2))))
print(V_B.pdf(np.zeros(2)))

### Goal: calc E(B.T f(V_B)).
print(B.transpose())
assert B.transpose().sample(5).shape != B.sample(5).shape
print(V_B.transpose().sample(5))

### Test the outer product for Gaussians
zz = Gaussian(np.random.rand(1, 3), np.eye(3))
outer = a.transpose() @ zz
# Have to implement (a is Gaussian) to be true
assert E(outer, num_samples = 10).shape == (2, 3) # Have to fix this to work with num=1 samples.
### Test empirical_cov
zz2 = Gaussian(np.random.rand(1, 3), np.eye(3))
outer = zz.transpose() @ zz2
assert E(outer, num_samples=5).shape == (3, 3)

### Test the outer product for Gaussian with a Compound Gaussian both with dependent and independent gaussians (TODO: outer prod of two compound gaussians)
outer = B.transpose() @ V_B
assert E(outer, num_samples = 5).shape == (2, 2)
outer = a.transpose() @ V_B
expectation = E(outer, num_samples = 50)
assert expectation.shape == (2, 2)
### Test it the other way around, should return similar
outer2 = V_B.transpose() @ a
expectation2 = E(outer2, num_samples = 50)
assert expectation2.shape == (2, 2)
### Test that two estimated covariance matrices from outer products of different permutations are the sample
g1 = Gaussian(100*np.ones((1, 2)), np.eye(2))
g2 = Gaussian(np.zeros((1, 2)), np.eye(2))
outer1 = g1.transpose() @ g2
outer2 = g2.transpose() @ g1
S1 = E(outer1, num_samples = 10**2)
S2 = E(outer2, num_samples = 10**2)
assert S1.shape == S2.shape
### Test it with zero mean, will use the numpy covariance function
g1 = Gaussian(np.zeros((1, 2)), np.eye(2))
g2 = Gaussian(np.zeros((1, 2)), np.eye(2))
outer1 = g1.transpose() @ g2
outer2 = g2.transpose() @ g1
S1 = E(outer1, num_samples = 10**5)
S2 = E(outer2, num_samples = 10**5)
assert np.linalg.norm(S1 - S2, ord = 'fro') < 10**(-2)

## Test the covariance function
assert cov(g1, g2, size = 10**5).shape == (2, 2)