using Plots

include("model.jl")
include("datasets.jl")

data = gen_synthetic(N=90, T=1000)

W, H, time_hist, loss_hist = fit_cnmf(data, L=10, K=5)

plot(time_hist, loss_hist, xlabel="Time", ylabel="Loss")
gui()
;
