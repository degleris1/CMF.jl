using Plots
using CMF: fit_cnmf, synthetic_sequences


data, W, H = synthetic_sequences(N=500, T=2000)

plot(xlabel="Time", ylabel="Loss")

alg_results = Dict()
for (alg, kwargs, label) in [
    [:hals, Dict(), "HALS"],
    [:mult, Dict(), "MULT"],
    #[:anls, Dict(), "ANLS"],
]
    results = fit_cnmf(data; L=20, K=3,
                       alg=alg, max_itr=Inf, max_time=30,
                       kwargs...
                       )

    plot!(results.time_hist, results.loss_hist, label=label)
    scatter!(results.time_hist, results.loss_hist, markersize=1, label="")

    alg_results[alg] = results
end

savefig("cnmf_alg_comparison.png")
gui()
;
