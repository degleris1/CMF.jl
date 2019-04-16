import JLD
import HDF5

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
                  l1_H=0, l2_H=0, l1_W=0, l2_W=0,
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
        if alg == :anls
            (l1_H == 0 &&
             l2_H == 0 &&
             l1_W == 0 &&
             l2_W == 0) || error("Regularization not supported with ANLS")
             loss, meta = ALGORITHMS[alg].update!(data, W, H, meta;
                                             kwargs...)
        else
            loss, meta = ALGORITHMS[alg].update!(data, W, H, meta;
                                                 l1_H=l1_H, l2_H=l2_H,
                                                 l1_W=l1_W, l2_W=l2_W,
                                                 kwargs...)
        end
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
    HDF5.h5open(path, "w") do file
        HDF5.write(file, "W", results.W)
        HDF5.write(file, "H", results.H)
        HDF5.write(file, "data", results.data)
        HDF5.write(file, "loss_hist", results.loss_hist)
        HDF5.write(file, "time_hist", results.time_hist)
    end
        
end
"""
Load a CNMF_results struct using JLD.
"""
function load_model(path)
    c = HDF5.h5open(path, "r") do file
        HDF5.read(file, "W")
        HDF5.read(file, "H")
        HDF5.read(file, "data")
        HDF5.read(file, "loss_hist")
        HDF5.read(file, "time_hist")
    end

    return CNMF_results(data, W, H, time_hist, loss_hist)
end
;
