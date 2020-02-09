"""
An abstract type for an update rule that updates both W and H.

Must implement:
    update!(rule::AbstractCFUpdate, data, W, H; kwargs...)
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

    W = deepcopy(W_init)
    H = deepcopy(H_init)

    # Set up optimization tracking
    loss_hist = [compute_loss(data, W, H)]
    time_hist = [0.0]

    itr = 1
    while (itr <= alg.max_itr) && (time_hist[end] <= alg.max_time) 
        itr += 1

        # Update with timing
        t0 = time()
        
        loss = update!(
            alg.update_rule, data, W, H; 
            kwargs...
        )
        dur = time() - t0
        
        # Record time and loss
        push!(time_hist, time_hist[end] + dur)
        push!(loss_hist, loss)

        # Check convergence
        if check_convergence && converged(loss_hist, patience, tol)
            break
        end
    end
end