using PyPlot; plt = PyPlot

include("../src/separable.jl")

# Generate data
data, tW, tH, (N, T, K, L) = generate_separable_data()

# Add noise
noisy_data = data + (0.01 * rand(N, T))


W, H = fit_conv_separable(noisy_data, K, L)
perm = permute_factors(tH, H)

plt.figure()
plt.imshow(H[perm, :], aspect="auto")
plt.title("Fit")
plt.show()

plt.figure()
plt.imshow(tH, aspect="auto")
plt.title("Truth")
plt.show()
