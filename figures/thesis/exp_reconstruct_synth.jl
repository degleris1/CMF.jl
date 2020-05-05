using Random
using MAT
using Distributions: Bernoulli, Uniform
using StatsBase

using Revise
using CMF

include("eval.jl")

folder = "/home/asd/data/thesis/"
Random.seed!(47)

N, T, K, L = 256, 10_000, 10, 30
sparsity = 0.25

# Generate motifs
w = zeros(K, N, L)
for k = 1:K
    for n = 1:N
        omega = rand(Uniform(0, 1))

        # Pick a random amplitude
        amplitude = rand(Uniform(1, 3)) * rand(Bernoulli(0.3))
        w[k, n, :] = abs.(amplitude * sin.(omega * (1:L)))

        # Pick a random dynamic
        dyn = rand()
        if dyn > 2/3 || k == 4
            w[k, 1, :] .*= (1:L) / L
        elseif dyn > 1/3
            w[k, 1, :] .*= (1 .- (1:L)/L)
        end
    end
end

# Generate feature maps
h = rand(Bernoulli(sparsity / K / L), K, T) .* rand(Uniform(1, 2), K, T)

# Generate observations
b = CMF.tensor_conv(w, h)
println("Generated data.")


# TODO Split into train and test sets 


# TODO Generate mask
#num_hidden = Int(0.1 * T)
#mask = ones(size(b))
#mask[sample(1:T, num_hidden, replace=false)] .= 0




# Fit model
λ = 0

println("Fitting...")
r = fit_cnmf(
    b; L=L+10, K=1,
    alg=PGDUpdate,
    max_itr=Inf,
    max_time=30,
    loss_func=CMF.SquareLoss(),
    tol=1e-5,
    penaltiesW=[],
    penaltiesH=[CMF.AbsolutePenalty(λ)],
    constrW=CMF.UnitNormConstraint(),
    seed=74,
)
println("Done.")

plt.figure()
plt.plot(r.time_hist, r.loss_hist)
;