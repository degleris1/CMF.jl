module MULT


# Imports
include("./common.jl")


function update(data, W, H, meta, options)
    if (meta == nothing)
        meta = _initialize_meta(data, W, H)
    end
    
    num_W, denom_W = _compute_mult_W(data, W, H)
    W .*= num_W ./ (denom_W .+ EPSILON)

    num_H, denom_H = _compute_mult_H(data, W, H)
    H .*= num_H ./ (denom_H .+ EPSILON)

    # Cache resids
    meta.resids = compute_resids(data, W, H)
    
    return norm(meta.resids) / meta.data_norm, meta
end


"""
Private
"""


mutable struct MultMeta
    resids
    data_norm
end


function _initialize_meta(data, W, H)
    resids = compute_resids(data, W, H)
    data_norm = norm(data)
    return MultMeta(resids, data_norm)
end


function _compute_mult_W(data, W, H)
    L, N, K = size(W)
    T = size(H)[2]

    num = zeros(L, N, K)
    denom = zeros(L, N, K)

    est = tensor_conv(W, H)
    
    for lag = 0:(L-1)
        num[lag+1, :, :] = data[:, 1+lag:T] * shift_cols(H, lag)'
        denom[lag+1, :, :] = data[:, 1+lag:T] * shift_cols(H, lag)'
    end

    return num, denom
end


function _compute_mult_H(data, W, H)
    est = tensor_conv(W, H)
    return tensor_transconv(W, data), tensor_transconv(W, est)
end


end  # module
;
