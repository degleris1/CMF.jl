"""
An abstract type for an update rule that updates both W and H.

Must implement:
    update!(rule::AbstractCFUpdate, data, W, H; kwargs...)
    UpdateRule(data, W, H)
"""
abstract type AbstractCFUpdate end

struct AlternatingOptimizer <: AbstractCFAlgorithm
    update_rule::AbstractCFUpdate
    max_itr
    max_time
end

function fit(
    alg::AlternatingOptimizer,
    data::Matrix{Float64},
    L::Int64,
    K::Int64,
    W_init::Tensor{Float64},
    H_init::Matrix{Float64};
    kwargs...
)
    # Load keyword args
    check_convergence = get(kwargs, :check_convergence, false)
    patience = get(kwargs, :patience, 0)
    tol = get(kwargs, :tol, 1e-4)

    W = deepcopy(W_init)
    H = deepcopy(H_init)

    # Set up optimization tracking
    loss_hist = [compute_loss(data, W, H)]
    time_hist = [0.0]

    datamean = sum(data) / length(data)

    itr = 1
    while (itr <= alg.max_itr) && (time_hist[end] <= alg.max_time) 
        itr += 1

        # Update with timing
        t0 = time()
        
        update_motifs!(alg.update_rule, data, W, H; kwargs...)
        loss = update_feature_maps!(alg.update_rule, data, W, H; kwargs...)

        # Normalize entries of W to mean of data
        #renormalize!(W, H, datamean)

        # Record time and loss
        dur = time() - t0
        push!(time_hist, time_hist[end] + dur)
        push!(loss_hist, loss)

        # Check convergence
        if check_convergence && converged(loss_hist, patience, tol)
            break
        end
    end

    return CNMF_results(data, W, H, time_hist, loss_hist)
end


function renormalize!(W::Tensor, H::Matrix, datamean)
    L, N, K = size(W)
    T = size(H, 2)

    # TODO renormalize so max of a row is 1
    for k in 1:K
        meanWk = sum(W[:, :, k]) / (L*N)

        W[:, :, k] ./= meanWk
        H[k, :] .*= meanWk
    end
end