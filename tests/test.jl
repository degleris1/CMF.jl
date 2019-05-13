using PyPlot; plt = PyPlot
using Revise
using CMF: fit_cnmf, synthetic_sequences, init_rand

K, L = 3, 10
data, W, H = synthetic_sequences(N=250, T=2500, K=K, L=L)
initW, initH = init_rand(data, L, K)


alg_results = Dict()
settings = [
    [:hals, Dict(), "HALS"],
    #[:mult, Dict(), "MULT"],
    [:anls, Dict(), "ANLS"]
]

plt.figure()
for (alg, kwargs, label) in settings
    results = fit_cnmf(data; L=L, K=K,
                       alg=alg, max_itr=Inf, max_time=5,
                       initW=initW, initH=initH,
                       kwargs...
                       )

    plt.plot(results.time_hist, results.loss_hist, label=label, marker=".")
    alg_results[alg] = results
end
plt.legend()
plt.xlabel("Time (seconds)")
plt.ylabel("Loss")

plt.savefig("cnmf_test.png")
;
