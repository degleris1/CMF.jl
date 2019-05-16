using PyPlot; plt = PyPlot

include("../src/separable.jl")

# Generate data
data, tW, tH, (N, T, K, L) = generate_separable_data()

# Add noise
noisy_data = data + (1 * rand(N, T))

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
