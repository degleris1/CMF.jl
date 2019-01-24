using Plots

include("model.jl")
include("datasets.jl")

data = gen_synthetic(N=500, T=2000)

plot(xlabel="Time", ylabel="Loss")

for trial in [
    #["hals", Dict(), "HALS"],
    ["mult", Dict(), "MULT"],
    ["hals", Dict("mode" => "elementwise"), "EW-HALS"]
]
    results = fit_cnmf(data, L=10, K=5,
                                          alg=trial[1], alg_options=trial[2],
                                          max_itr=1000, max_time=30)
    time_hist = results.time_hist
    loss_hist = results.loss_hist
    plot!(time_hist, loss_hist, label=trial[3])
    scatter!(time_hist, loss_hist, markersize=1, label="")
    savefig("cnmf_alg_comparison.png")
end
    
gui()
;
