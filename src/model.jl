include("./mult.jl")  # MultUpdate
include("./hals.jl")  # HALSUpdate
include("./annls.jl") 
include("./common.jl")


ALGORITHMS = Dict(
    "mult" => MULT,
    "hals" => HALS,
    "annls" => ANNLS
)

struct CNMF_results
    data::Array{Float64}
    W::Array{Float64}
    H::Array{Float64}
    time_hist::Array{Float64}
    loss_hist::Array{Float64}
    function CNMF_results(data, W, H, time_hist, loss_hist)
        return new(data, W, H, time_hist, loss_hist)
    end
end


function fit_cnmf(data; L=7, K=3, alg="mult",
                  alg_options=Dict(), max_itr=100, max_time=Inf)
    # Initialize
    W, H = init_rand(data, L, K)
    meta = nothing
    
    # Set up optimization tracking
    loss_hist = [compute_loss(data, W, H)]
    time_hist = [0.0]

    # Update
    itr = 1
    tot_time = 0
    while (itr <= max_itr) && (time_hist[end] <= max_time) 
        itr += 1

        # Update with timing
        t0 = time()
        loss, meta = ALGORITHMS[alg].update(data, W, H, meta, alg_options)
        dur = time() - t0
        
        # Record time and loss
        push!(time_hist, time_hist[end] + dur)
        push!(loss_hist, loss)
    end

    return CNMF_results(data, W, H, time_hist, loss_hist)
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
