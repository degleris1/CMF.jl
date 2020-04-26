using Random
using MAT
using Distributions: Bernoulli, Uniform
using StatsBase

using Revise
using CMF

include("eval.jl")

folder = "/home/asd/data/thesis/"
Random.seed!(7)

N = 1
T, K, L = 100_000, 5, 100

# Generate motifs
w = zeros(K, N, L)
for k = 1:K
    # Pick a random frequency
    omega = rand(Uniform(0, 1))

    # Pick a random amplitude
    amplitude= rand(Uniform(1, 3))
    w[k, 1, :] = amplitude * sin.(omega * (1:L))

    # Pick a random dynamic
    dyn = rand()
    if dyn > 2/3 || k == 4
        w[k, 1, :] .*= (1:L) / L
    elseif dyn > 1/3
        w[k, 1, :] .*= (1 .- (1:L)/L)
    end
end

# Generate feature maps
sparsity = 0.25 / K / L
h = rand(Bernoulli(sparsity), K, T) .* rand(Uniform(1, 2), K, T)

# Generate observations
b = CMF.tensor_conv(w, h)

# Generate mask
num_hidden = Int(0.1 * T)
mask = ones(size(b))
mask[sample(1:T, num_hidden, replace=false)] .= 0


l1min = 0.5
l1max = 100
l1weights = exp.(range(log(l1min), log(l1max), length=30))
results = []
trainerrs = []
testerrs = []
nnzs = []

relweights = range(0.01, 0.99, length=20)
α = 0.6
initW = randn(K, N, L)
initH = rand(K, T)

# Fit model
λ = 10*maximum(b)
θ = 2*maximum(b)
#for λ in [10*maximum(b)]  #l1weights
    push!(results, [])
    push!(trainerrs, [])
    push!(testerrs, [])
    push!(nnzs, [])

    #for θ in [maximum(b)]  #relweights

        @show λ, θ

        @time r = fit_cnmf(
            b; L=L, K=K,
            alg=PGDUpdate,
            max_itr=Inf,
            max_time=60,
            loss_func=CMF.MaskedLoss(CMF.SquareLoss(), mask),
            tol=1e-5,
            penaltiesW=[CMF.SquarePenalty(θ)],
            penaltiesH=[CMF.AbsolutePenalty(λ)],
            constrW=nothing,
            W_init=(1-α)*w + α*initW,
            H_init=(1-α)*h + α*initH,
        )

        # Visualize motifs
        # plt.close()
        plt.figure()
        for k = 1:K
            plt.subplot(K, 2, 2*k-1)
            plt.plot(1:2L, [w[k, 1, :]; zeros(L)])  # / maximum(w[k, 1, :]))
            plt.xticks([])
            #plt.yticks([])
            plt.grid()

            plt.subplot(K, 2, 2*k)
            plt.plot(1:2L, [r.W[k, 1, :]; zeros(L)])  # / maximum(r.W[k, 1, :]))
            plt.xticks([])
            #plt.yticks([])
            plt.grid()
        end

        est = CMF.tensor_conv(r.W, r.H)
        testmask = 1 .- mask
        train_error = norm(mask .* (est - b)) / norm(mask .* b)
        test_error = norm(testmask .* (est - b)) / norm(testmask .* b)
        nnz_est = sum(r.H .> 0.1)
        nnz_true = sum(h .> 0.001)

        @show train_error, test_error
        @show nnz_est / nnz_true
        println()

        push!(results[end], r)
        push!(trainerrs[end], train_error)
        push!(testerrs[end], test_error)
        push!(nnzs[end], nnz_est)
    #end
#end

# # Plot 
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(l1weights, trainerrs, label="train")
# plt.plot(l1weights, testerrs, label="test")
# plt.xscale("log")
# plt.yscale("log")
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(l1weights, nnzs, label="nnz")
# plt.hlines(sum(h .> 0.001), l1min, l1max)
# plt.xscale("log")
# plt.legend()

# Save results
;