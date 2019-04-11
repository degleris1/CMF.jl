import PyPlot; const plt = PyPlot
import CMF; const cmf = CMF

data = cmf.gen_synthetic()

println("Fitting models...")
hals_result = cmf.fit_cnmf(data, alg=:hals, max_time=10, max_itr=Inf)
mult_result = cmf.fit_cnmf(data, alg=:mult, max_time=10, max_itr=Inf)

fig, ax = plt.subplots(1, 1)
ax.plot(hals_result.time_hist, hals_result.loss_hist, label="HALS")
ax.plot(mult_result.time_hist, mult_result.loss_hist, label="MU")
ax.legend()

# First figure, showing different methods.
fig, ax = plt.subplots(3, 1, sharex=true)

ax[1].imshow(data)
ax[1].set_title("Raw Data")

ax[2].imshow(cmf.tensor_conv(mult_result.W, mult_result.H))
ax[2].set_title("Multiplicative Updates")

ax[3].imshow(cmf.tensor_conv(hals_result.W, hals_result.H))
ax[3].set_title("HALS")
plt.tight_layout()


# Second figure, showing
fig = cmf.plot_result(hals_result)
