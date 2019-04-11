using JLD

include("./mult.jl")  # MultUpdate
include("./hals.jl")  # HALSUpdate
include("./anls.jl") 
include("./common.jl")


ALGORITHMS = Dict(
    :mult => MULT,
    :hals => HALS,
    :anls => ANLS
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


function fit_cnmf(data; L=10, K=5, alg=:mult,
                  max_itr=100, max_time=Inf,
                  kwargs...)
    # Initialize
    W, H = init_rand(data, L, K)
    W = get(kwargs, :initW, W)
    H = get(kwargs, :initH, H)
    
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
        loss, meta = ALGORITHMS[alg].update!(data, W, H, meta; kwargs...)
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


"""
Fit several models with varying parameters.
Possible to iterate over lags (L), number of components (K), and different algorithms (alg).
"""
function parameter_sweep(data; L_vals=[7], K_vals=[3], alg_vals=[:mult],
                         alg_options=Dict(), max_itr=100, max_time=Inf,
                         lambda1=0, lambda2=0, initW=nothing, initH=nothing)
    all_results = Dict()
    for (L, K, alg) in Iterators.product(L_vals, K_vals, alg_vals)
        all_results[(L, K, alg)] = fit_cnmf(
            data, L=L, K=K, alg=alg,
            alg_options=alg_options, max_itr=max_itr, max_time=max_time,
            lambda1=lambda1, lambda2=lambda2, initW=initW, initH=initH
        )
    end

    return all_results
end



"""
Simple wrapper to save a CNMF_results struct using JLD.
"""
function save_model(results::CNMF_results, path)
    jldopen(path, "w") do file
        write(file, "data", results.data)
        write(file, "W", results.W)
        write(file, "H", results.H)
        write(file, "time_hist", results.time_hist)
        write(file, "loss_hist", results.loss_hist)
    end
end

"""
Load a CNMF_results struct using JLD.
"""
function load_model(path)
    results_dict = JLD.load(path)
    data = results_dict["data"]
    W = results_dict["W"]
    H = results_dict["H"]
    time_hist = results_dict["time_hist"]
    loss_hist = results_dict["loss_hist"]
    return CNMF_results(data, W, H, time_hist, loss_hist)
end
;
