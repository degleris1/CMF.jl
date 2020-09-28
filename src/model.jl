@mlj_model mutable struct ConvolutionalFactorization
    L::Int = 1::(_ > 0)
    K::Int = 1::(_ > 0)
    loss::AbstractLoss = SquareLoss()
    W_penalizers::Vector{AbstractPenalty} = AbstractPenalty[]
    H_penalizers::Vector{AbstractPenalty} = AbstractPenalty[]
    W_constraint::AbstractConstraint = NoConstraint()
    H_constraint::AbstractConstraint = NoConstraint()
    algorithm::Symbol = :pgd::(_ in (:pgd,))
    max_iters::Int = 100::(_ > 0)
    max_time::Float64 = Inf::(_ > 0)
end

function eval(
    model::ConvolutionalFactorization,
    W,
    H,
    b,
    est
)
    return eval(
        model.loss,
        model.W_penalizers,
        model.H_penalizers,
        W,
        H,
        b,
        est)
end

function eval(
    model::ConvolutionalFactorization,
    W,
    H,
    b,
)
    return eval(
        model.loss,
        model.W_penalizers,
        model.H_penalizers,
        W,
        H,
        b,
        tensor_conv(W,H))
end


function MLJModelInterface.fit(
    model::ConvolutionalFactorization, 
    verbosity::Int, 
    X;
    seed::Int = 123 
)
    (seed !== nothing) && Random.seed!(seed)

    # Initialize
    W_init, H_init = init_rand(X, model.L, model.K)

    # Set up alternating optimizer (to be generalized)
    update_rule = Dict(
        :pgd => PGDUpdate
    )[model.algorithm]

    alg = AlternatingOptimizer(
        update_rule(X, W_init, H_init),
        model.max_iters,
        model.max_time
    )

    W, H, time_hist, loss_hist = _fit(
        model, alg, X, W_init, H_init,
        verbose=(verbosity >= 2),
    )

    fitresult = (W=W, H=H)
    cache = nothing
    report = (time=time_hist, loss=loss_hist)

    return fitresult, cache, report
end


function MLJModelInterface.update(
    model::ConvolutionalFactorization, 
    verbosity::Int,
    old_fitresult,
    old_cache,
    X
)
end


function MLJModelInterface.transform(
    model::ConvolutionalFactorization,
    fitresult,
    Xnew;
    seed=123
)
    (seed !== nothing) && Random.seed!(seed)

    # Initialize
    W_init, H_init = init_rand(X, model.L, model.K)
    W_init = fitresult.W

    # Set up alternating optimizer (to be generalized)
    update_rule = Dict(
        :pgd => PGDUpdate
    )[model.algorithm]

    alg = AlternatingOptimizer(
        update_rule(X, W_init, H_init),
        model.max_iters,
        model.max_time
    )

    # Given a new X and a W, it finds the best H to fit the data
    W, H, time_hist, loss_hist = fit(
        alg, Xnew, model.L, model.K, W_init, H_init, evalmode=true
    )
end


####        MOVE STUFF BELOW HERE


"""
Check for model convergence
"""
function converged(loss_hist, patience, tol)

    # If we have not run for `patience` iterations,
    # we have not converged.
    if length(loss_hist) <= patience
        return false
    end

    d_loss = diff(loss_hist[end-patience:end])

    # Objective converged
    if (all(abs.(d_loss) .< tol))
        return true
    else
        return false
    end
end


"""
Initialize randomly, scaling to minimize square error.
"""
function init_rand(data, L, K)
    N, T = size(data)

    W = rand(K, N, L)
    H = rand(K, T)

    est = tensor_conv(W, H)
    alpha = (reshape(data, N*T)' * reshape(est, N*T)) / norm(est)^2
    W *= sqrt(abs(alpha))
    H *= sqrt(abs(alpha))

    return W, H
end


# """Sorts units to reveal sequences."""
# function sortperm(r::CNMF_results)
#     W_norm = zeros(size(r.W))
#     for k in 1:size(r.W, 1)
#         W_norm[k, :, :] /= norm(r.W[k, :, :])
#     end
    
#     # For each unit, compute the largest weight across
#     # components.
#     sum_over_lags = dropdims(sum(W_norm, dims=1), dims=1)
#     rows = [view(sum_over_lags, i, :) for i in axes(sum_over_lags, 1)]
#     max_component = argmax.(rows)

#     # For each unit, compute the largest weight across
#     # lags (within largest component).
#     max_lag = Int64[]
#     for (n, c) in enumerate(max_component)
#         push!(max_lag, argmax(W_norm[:, n, c]))
#     end

#     # Lexographically sort units.
#     return sortperm(
#         [CartesianIndex(i, j) for (i, j) in zip(max_lag, max_component)])
# end