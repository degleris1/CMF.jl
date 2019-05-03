using Plots
using Revise
using CMF: fit_cnmf, synthetic_sequences, init_rand

K, L = 3, 10
data, W, H = synthetic_sequences(N=100, T=250, K=K, L=L)
initW, initH = init_rand(data, L, K)

plot(xlabel="Time", ylabel="Loss")

alg_results = Dict()
settings = [
    [:hals, Dict(), "HALS"],
    #[:mult, Dict(), "MULT"],
    [:anls, Dict(), "ANLS"],
    [:anls, Dict(:variant => :cache), "Cached ANLS"],
    [:anls, Dict(:variant => :pivot), "Pivot ANLS"],
]

for (alg, kwargs, label) in settings
    results = fit_cnmf(data; L=L, K=K,
                       alg=alg, max_itr=Inf, max_time=5,
                       initW=initW, initH=initH,
                       kwargs...
                       )

    plot!(results.time_hist, results.loss_hist, label=label)
    scatter!(results.time_hist, results.loss_hist, markersize=1, label="")

    alg_results[alg] = results
end

savefig("cnmf_test.png")
gui()
;
