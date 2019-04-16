module MULT


# Imports
include("./common.jl")


function update!(data, W, H, meta; kwargs...)

    # unpack regularization args
    l1_W = get(kwargs, :l1_W, 0)
    l1_H = get(kwargs, :l1_H, 0)
    l2_W = get(kwargs, :l2_W, 0)
    l2_H = get(kwargs, :l2_H, 0)

    if (meta == nothing)
        meta = MultMeta(data, W, H)
    end
    
    num_W, denom_W = _compute_mult_W(data, W, H)
    W .*= num_W ./ (denom_W .+ l1_W .+ 2 .* l2_W .* W .+ EPSILON)

    num_H, denom_H = _compute_mult_H(data, W, H)
    H .*= num_H ./ (denom_H .+ l1_H .+ 2 .* l2_H .* H .+ EPSILON)

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
    function MultMeta(data, W, H)
        resids = compute_resids(data, W, H)
        data_norm = norm(data)
        return new(resids, data_norm)
    end
end


function _compute_mult_W(data, W, H)
    L, N, K = size(W)
    T = size(H)[2]

    num = zeros(L, N, K)
    denom = zeros(L, N, K)

    est = tensor_conv(W, H)
    
    for lag = 0:(L-1)
        num[lag+1, :, :] = data[:, 1+lag:T] * shift_cols(H, lag)'
        denom[lag+1, :, :] = est[:, 1+lag:T] * shift_cols(H, lag)'
    end

    return num, denom
end


function _compute_mult_H(data, W, H)
    est = tensor_conv(W, H)
    return tensor_transconv(W, data), tensor_transconv(W, est)
end


end  # module
;
