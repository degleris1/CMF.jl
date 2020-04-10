using Random
using MAT

using Revise
using CMF

include("../../datasets/synthetic.jl")
include("./eval.jl")

folder = "/home/asd/data/thesis/"
Random.seed!(7)

N, T, K, L = 50, 2000, 5, 30
ntrials = 30
num_noise = 20
noiselevels = exp.( range(log(0.1), log(10), length=num_noise) )

for trial = 1:ntrials
    data, W, H = synthetic_sequences(N=N, T=T, K=K, L=L, noise_scale=0, sigma=0.1)
    noise = randn(N, T)
    initW, initH = CMF.init_rand(data, 2L, K)

    for n in 1:num_noise
        println("Noise=", noiselevels[n])
        noisydata = max.(0, data + noiselevels[n] * noise)

        @time r = fit_cnmf(
            noisydata; L=2L, K=K,
            alg=PGDUpdate, 
            max_itr=Inf, 
            max_time=120, 
            tol=1e-4,
            penaltiesW=[CMF.SquarePenalty(N*T)],
            penaltiesH=[CMF.AbsolutePenalty(1)],
            initW=initW,
            initH=initH,
        )

        score, perm, offsets, est = evalW(r.W, W)

        @show r.loss_hist[end]
        @show score
        println()

        matwrite(
            string(folder, "latsyn/", noise, "_", trial, ".mat"),
            Dict(
                "data" => data,
                "trueW" => W,
                "trueH" => H,
                "loss_hist" => r.loss_hist,
                "time_hist" => r.time_hist,
                "estW" => r.W,
                "estH" => r.H,
            )
        )
    end
    println("---")
    println("")
end