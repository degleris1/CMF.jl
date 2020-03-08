using Revise
using CMF
using Random

import PyPlot; plt = PyPlot

include("../datasets/synthetic.jl")

Random.seed!(1234)

K, L = 3, 7
data, W, H = synthetic_sequences(N=100, T=5000, K=K, L=L)
initW, initH = init_rand(data, L, K)


alg_results = Dict()
settings = [
    [ADMMUpdate, Dict(), "ADMM-10-15"],
    #[ADMMUpdate, Dict(:admm_H_maxiter => 10), "ADMM-10-10"],
    #[ADMMUpdate, Dict(:admm_W_maxiter => 15), "ADMM-15-15"],
    #[HALSUpdate, Dict(), "HALS"],
    #[CMF.MultUpdate, Dict(), "MULT"],
    [CMF.ANLSUpdate, Dict(), "ANLS"]
]

#plt.figure()
for (alg, kwargs, label) in settings
    println("Testing ", label)
    results = fit_cnmf(
        data; L=L, K=K,
        alg=alg, max_itr=Inf, max_time=30,
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
