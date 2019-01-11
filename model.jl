include("mult.jl")
include("common.jl")

function fit_cnmf(data; L=7, K=3, max_itr=100, update=update_mult)
    # Initialize
    W, H = init_rand(data, L, K)
    meta = nothing
    
    # Set up optimization tracking
    loss_hist = [compute_loss(data, W, H)]
    time_hist = [0.0]

    # Update
    for itr = 1:max_itr
        # Update with timing
        t0 = time()
        loss, meta = update(data, W, H, meta)
        dur = time() - t0

        # Record time and loss
        push!(time_hist, time_hist[end] + dur)
        push!(loss_hist, loss)
    end

    return W, H, time_hist, loss_hist
end


"""
Initialize randomly, scaling to minimize square error.
"""
function init_rand(data, L, K)
    N, T = size(data)

    W = rand(L, N, K)
    H = rand(K, T)

    est = tensor_conv(W, H)
    alpha = (reshape(data, N*T)' * reshape(est, N*T)) / norm(est)^2
    W *= sqrt(alpha)
    H *= sqrt(alpha)

    return W, H
end
;
