using Revise
using CMF
using Random

import PyPlot; plt = PyPlot

include("../datasets/synthetic.jl")

Random.seed!(1234)

# 100, 1000, 10, 50 --- ADMM takes 27 seconds, 19 iterations
# Goal: 100, 100_000, 50, 100

N, T, K, L = 100, 250, 10, 20
data, W, H = synthetic_sequences(N=N, T=T, K=K, L=L)
initW, initH = init_rand(data, L, K)


alg_results = Dict()
settings = [
    [HALSUpdate, Dict(), "HALS"],
    [MultUpdate, Dict(), "MULT"],
    [PGDUpdate, Dict(), "PGD"],
    [ADMMUpdate, Dict(), "ADMM"],
]

plt.figure()
for (alg, kwargs, label) in settings
    println("Testing ", label)
    @time results = fit_cnmf(
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
plt.ylabel("Log Loss")
plt.yscale("Log")

# plt.savefig("cnmf_test.png")
;
