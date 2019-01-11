using LinearAlgebra
include("./common.jl")

function update_hals(data, W, H, meta)
    if (meta == nothing)
        meta = _initialize_meta(data, W, H)
    end

    # Update W
    _setup_W_update(W, H, meta)
    _update_W(W, meta.H_unfold, meta.H_norms, meta.resids)

    
    return norm(resids) / data_norm, meta
end



"""
Initial update
"""
mutable struct HALSMeta
    resids  # Internals
    data_norm
    batch_inds
    batch_sizes
    
    H_unfold  # W setup
    H_norms
end


function _intialize_meta(data, W, H)
    resids = tensor_conv(W, H) - data
    data_norm = norm(data)

    # Set up batches
    batch_inds = []
    batch_sizes = []
    
    return HALSMeta(resids, data_norm, batch_inds, batch_sizes,  # Internals
                    nothing, nothing)  # W setup
end



"""
W updates
"""


function _update_W(W, H_unfold, H_norms, resids)
    L, N, K = size(W)
    for k = 1:K
        for l = 1:L
            _update_W_col(k, l, W, H_unfold, H_norms, resids)
        end
    end
end


function _setup_W_update(W, H, meta)
    L, N, K = size(W)

    meta.H_unfold = shift_and_stack(H, L)  # Unfold matrices
    meta.H_norms = zeros(K*L)  # Compute norms
    for i=1:(K*L)
        meta.H_norms[i] = norm(H_unfold[i, :])
    end
end


function _update_W_col(k, l, W, H_unfold, H_norms, resids)
    L, N, K = size(W)
    ind = l*K + k

    resids .-= W[l, :, k] * H_unfold[ind, :]'  # outer product
    W[l, :, k] = _next_W_col(H_unfold[ind, :], H_norms[ind], resids)
    resids .+= W[l, :, k] * H_unfold[ind, :]'  # outer product
end

                             
function _next_W_col(Hkl, norm_Hkl, resid)
    return max.((-resid * Hkl) ./ (norm_Hkl^2 + EPSILON), 0.0)
end
;
