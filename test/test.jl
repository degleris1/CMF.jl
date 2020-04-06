using Revise
using CMF
using Random

import PyPlot; plt = PyPlot

include("../datasets/synthetic.jl")

Random.seed!(1234)

K, L = 3, 7
data, W, H = synthetic_sequences(N=100, T=500, K=K, L=L)
initW, initH = init_rand(data, L, K)


alg_results = Dict()
settings = [
    [ADMMUpdate, Dict(), "ADMM"],
    #[ADMMUpdate, Dict(:fast => true), "ADMM-fast"],
    [HALSUpdate, Dict(), "HALS"],
    #[MultUpdate, Dict(), "MULT"],
    #[ANLSUpdate, Dict(), "ANLS"]
]

plt.figure()
for (alg, kwargs, label) in settings
    println("Testing ", label)
    results = fit_cnmf(
        data; L=L, K=K,
        alg=alg, max_itr=Inf, max_time=3,
        initW=initW, initH=initH,
        kwargs...
    )

    plt.plot(results.time_hist, results.loss_hist, label=label, marker=".")
    alg_results[alg] = results
end
plt.legend()
plt.xlabel("Time (seconds)")
plt.ylabel("Loss")

# plt.savefig("cnmf_test.png")
;
