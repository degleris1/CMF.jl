mutable struct MultUpdate <: AbstractCFUpdate
    resids
    data_norm
end

function MultUpdate(data, W, H)
    resids = compute_resids(data, W, H)
    data_norm = norm(data)
    return MultUpdate(resids, data_norm)
end


function update_motifs!(rule::MultUpdate, data, W, H; l1W=0, l2W=0, kwargs...)
    num_W, denom_W = _compute_mult_W(data, W, H)
    @. W *= num_W / (denom_W + l1W + 2*l2W*W + EPSILON)
end


function update_feature_maps!(rule::MultUpdate, data, W, H; l1H=0, l2H=0, kwargs...)
    num_H, denom_H = _compute_mult_H(data, W, H)
    H .*= num_H ./ (denom_H .+ l1H .+ 2 .* l2H .* H .+ EPSILON)

    rule.resids = compute_resids(data, W, H) 
    return norm(rule.resids) / rule.data_norm
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