using PyPlot; plt = PyPlot

include("../src/separable.jl")
using Main.Separable; sep = Main.Separable

# Generate data
N, T, K, L = 100, 150, 3, 5
data, tW, tH, = gen_sep_data(N, T, K, L)

# Add noise
noisy_data = data + (0.01 * rand(N, T))

W, H = fit_conv_separable(noisy_data, K, L)
perm = permute_factors(tH, H)

plt.figure()
plt.imshow(row_normalize(H[perm, :]), aspect="auto")
plt.title("Fit")
plt.show()

plt.figure()
plt.imshow(row_normalize(tH), aspect="auto")
plt.title("Truth")
plt.show()
