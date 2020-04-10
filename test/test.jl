using Revise
using CMF
using Random

import PyPlot; plt = PyPlot

include("../datasets/synthetic.jl")

Random.seed!(1234)

# 100, 1000, 10, 50 --- ADMM takes 27 seconds, 19 iterations
# Goal: 100, 100_000, 50, 100

N, T, K, L = 100, 100, 10, 5
data, W, H = synthetic_sequences(N=N, T=T, K=K, L=L)
initW, initH = init_rand(data, L, K)


alg_results = Dict()
rhoh = 40 / sqrt(maximum(data))
rhow = 10
settings = [
    #[HALSUpdate, Dict(), "HALS"],
    #[MultUpdate, Dict(), "MULT"],
    [PGDUpdate, Dict(), "PGD"],
    #[ADMMUpdate, Dict(:rhoh => rhoh, :rhow => rhow), "ADMM"],
]

plt.figure()
plt.xlabel("Time (seconds)")
plt.ylabel("Log Loss")
plt.yscale("Log")
plt.ylim([0.1, 1.1])
plt.grid(which="minor")

for (alg, kwargs, label) in settings
    println("Testing ", label)
    @time results = fit_cnmf(
        data; L=L, K=K,
        alg=alg, max_itr=Inf, max_time=5, tol=1e-4,
        initW=initW, initH=initH,
        kwargs...
    )

    plt.plot(results.time_hist, results.loss_hist, label=label, marker=".")
    alg_results[alg] = results
end
plt.legend()

# plt.savefig("cnmf_test.png")
;
