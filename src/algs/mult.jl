mutable struct MultUpdate <: AbstractCFUpdate
    resids
    data_norm
    numW
    denomW
    numH
    denomH
    est
end

function MultUpdate(data, W, H)
    resids = compute_resids(data, W, H)
    data_norm = norm(data)
    numW = zeros(size(W))
    denomW = zeros(size(W))
    numH = zeros(size(H))
    denomH = zeros(size(H))
    est = zeros(size(data))
    return MultUpdate(resids, data_norm, numW, denomW, numH, denomH, est)
end


function update_motifs!(rule::MultUpdate, data, W, H; l1W=0, l2W=0, kwargs...)
    K, N, L = size(W)
    T = size(H, 2)
   
    # Compute estimate
    tensor_conv!(rule.est, W, H)    

    # Compute numerator and denominator
    for lag = 0:(L-1)
        @views mul!(rule.numW[:, :, lag+1], shift_cols(H, lag), data[:, 1+lag:T]')
        @views mul!(rule.denomW[:, :, lag+1], shift_cols(H, lag), rule.est[:, 1+lag:T]')
    end

    # Update W
    @. W *= rule.numW / (rule.denomW + l1W + 2*l2W*W + eps())
    @. W = max(eps(), W)  # avoid zero-locking
end


function update_feature_maps!(rule::MultUpdate, data, W, H; l1H=0, l2H=0, kwargs...)
    # Update estimate
    tensor_conv!(rule.est, W, H)

    # Compute numerator and denominator
    tensor_transconv!(rule.numH, W, data)
    tensor_transconv!(rule.denomH, W, rule.est)

    # Update H
    @. H *= rule.numH / (rule.denomH + l1H + 2*l2H*H + eps())
    @. H = max(eps(), H)  # avoid zero-locking

    # Compute residual
    tensor_conv!(rule.est, W, H)
    @. rule.resids = rule.est - data
    return norm(rule.resids) / rule.data_norm
end