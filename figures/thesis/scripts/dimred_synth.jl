using Random
using MAT
using Distributions: Bernoulli, Uniform
using CMF

# Load command line args
model_K = parse(Int64, ARGS[1])
model_L = parse(Int64, ARGS[2])
savepath = ARGS[3]


# Generate data
println("Generating data...")
Random.seed!(47)
N, T, K, L = 256, 10_00, 7, 30
sparsity = 0.50

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


# Split into train and test sets 
split_point = 900
train = b[:, 1:split_point]
test = b[:, split_point+1:end]

# Fit model
println("Fitting with K=", model_K, ", L=", model_L, "...")
r = fit_cnmf(
    train; L=model_L, K=model_K,
    alg=PGDUpdate,
    max_itr=200,
    max_time=Inf,
    loss_func=CMF.SquareLoss(),
    check_convergence=false,
    penaltiesW=[CMF.SquarePenalty(1e-3)],
    penaltiesH=[CMF.AbsolutePenalty(1e-3)],
    constrW=CMF.NonnegConstraint(),
    constrH=CMF.NonnegConstraint(),
    seed=74,
)
@show r.loss_hist[end]
@show r.time_hist[end]
@show length(r.time_hist)

# Evaluate test data 
println("Evaluating test set...")
r_test = fit_cnmf(
    test; L=model_L, K=model_K,
    alg=PGDUpdate,
    max_itr=200,
    max_time=Inf,
    loss_func=CMF.SquareLoss(),
    check_convergence=false,
    penaltiesW=[CMF.SquarePenalty(1e-3)],
    penaltiesH=[CMF.AbsolutePenalty(1e-3)],
    constrW=CMF.NonnegConstraint(),
    constrH=CMF.NonnegConstraint(),
    W_init=r.W,
    seed=747,
    eval_mode=true,
)
@show r_test.loss_hist[end]
@show r_test.time_hist[end]
@show length(r_test.time_hist)




# Save results
println("Saving results...")
matwrite(
    string(savepath, "synth", model_L, "_", model_K, "full.mat"),
    Dict(
        "W" => r.W,
        "H" => r.H,
        "testH" => r_test.H,
    )
)
matwrite(
    string(savepath, "synth", model_L, "_", model_K, "sum.mat"),
    Dict(
        "train_loss" => r.loss_hist,
        "train_time" => r.time_hist,
        "test_loss" => r_test.loss_hist,
        "test_time" => r_test.time_hist,
    )
)