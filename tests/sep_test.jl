import PyPlot; plt = PyPlot

using Revise
using CMF
sep = CMF.Separable

include("../datasets/sep.jl")

# Generate data
N, T, K, L = 100, 250, 3, 5
data, tW, tH = gen_sep_data(N, T, K, L)

# Add noise
noise_level = 0.1
noisy_data = data + (noise_level * rand(N, T))

results = CMF.fit_cnmf(noisy_data, K=K, L=L, alg=:sep, thresh=0.2*N-noise_level)
W, H = results.W, results.H
perm = sep.permute_factors(tH, H)


plt.figure()
plt.imshow(sep.row_normalize(H[perm, :]), aspect="auto")
plt.title("Fit")
plt.show()

plt.figure()
plt.imshow(sep.row_normalize(tH), aspect="auto")
plt.title("Truth")
plt.show()

println("Score: ", sep.cos_score(H[perm, :], tH))

