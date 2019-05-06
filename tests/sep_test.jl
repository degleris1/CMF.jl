using PyPlot; plt = PyPlot

include("../src/separable.jl")


data, tW, tH, (N, T, K, L) = generate_separable_data()

W, H = fit_conv_separable(data, K, L)


plt.figure()
plt.imshow(H, aspect="auto")
plt.title("Fit")
plt.show()

plt.figure()
plt.imshow(tH, aspect="auto")
plt.title("Truth")
plt.show()
