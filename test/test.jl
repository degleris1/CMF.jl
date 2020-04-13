using PyPlot
using Random

using Revise
using CMF


include("../datasets/synthetic.jl")

Random.seed!(1234)

# 100, 1000, 10, 50 --- ADMM takes 27 seconds, 19 iterations
# Goal: 100, 100_000, 50, 100

N, T, K, L = 300, 1000, 10, 5
data, W, H = synthetic_sequences(N=N, T=T, K=K, L=L)
initW, initH = init_rand(data, L, K)
mask = zeros(size(data))
mask[:, :] .= 1


alg_results = Dict()
rhoh = 40 / sqrt(maximum(data))
rhow = 10
settings = [
    #[HALSUpdate, Dict(), "HALS"],
    #[MultUpdate, Dict(), "MULT"],
    [PGDUpdate, Dict(:extrapolate => true), "Accelerated PGD"],
    [PGDUpdate, Dict(:extrapolate => false), "PGD"],
    #[ADMMUpdate, Dict(:rhoh => rhoh, :rhow => rhow), "ADMM"],
]

plt.figure()
plt.xlabel("Time (seconds)")
plt.ylabel("Log Loss")
plt.yscale("Log")
plt.grid(which="minor")

min_loss = Inf
for (alg, kwargs, label) in settings
    println("Testing ", label)

    results = fit_cnmf(
        data; L=L, K=K,
        alg=alg, max_itr=Inf, max_time=10, tol=1e-4,
        check_convergence=false,
        initW=initW, initH=initH,
        loss_func=CMF.MaskedLoss(CMF.SquareLoss(), mask),
        kwargs...
    )
    alg_results[label] = results

    global min_loss = min(min_loss, results.loss_hist[end])
end


for (alg, kwargs, label) in settings
    plt.plot(alg_results[label].time_hist, alg_results[label].loss_hist .- min_loss, 
            label=label, marker=".")
end

plt.legend()

plt.savefig("cnmf_test.png")
;
