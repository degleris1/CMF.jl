# TODO
# * Reintroduce strided H updates

mutable struct HALSUpdate <: AbstractCFUpdate
    resids  # Internals
    data_norm
    
    H_unfold  # W setup
    H_norms

    W_norms  # H setup
    Wk_list
end


function HALSUpdate(data, W, H)
    L, N, K = size(W)
    T = size(H)[2]

    resids = tensor_conv(W, H) - data
    data_norm = norm(data)
    
    return HALSUpdate(resids, data_norm,  # Internals
                      nothing, nothing,  # W setup
                      nothing, nothing)  # H setup
end


function update!(
    rule::HALSUpdate, data, W, H; 
    l1_H=0, l2_H=0, l1_W=0, l2_W=0, kwargs...
)

    # W update
    _setup_W_update!(rule, W, H)
    _update_W!(W, rule.H_unfold, rule.H_norms, rule.resids, l1_W, l2_W)

    # H update
    _setup_H_update!(rule, W, H)
    _update_H_regular!(W, H, rule.resids, rule.Wk_list, rule.W_norms, l1_H, l2_H)
    
    return norm(rule.resids) / rule.data_norm
end



"""
-----------------------------------
Internals, Initialization and Setup
-----------------------------------
"""


function _setup_W_update!(rule::HALSUpdate, W, H)
    L, N, K = size(W)

    rule.H_unfold = shift_and_stack(H, L)  # Unfold matrices
    rule.H_norms = zeros(K*L)  # Compute norms
    for i=1:(K*L)
        rule.H_norms[i] = norm(rule.H_unfold[i, :])
    end
end


function _setup_H_update!(rule::HALSUpdate, W, H)
    L, N, K = size(W)
    
    # Set up norms
    rule.W_norms = zeros(K, L)
    for k = 1:K
        for l = 1:L
            rule.W_norms[k, l] = norm(W[l, :, k])
        end
    end
                                    
    # Slice W by k
    rule.Wk_list = []
    for k = 1:K
        push!(rule.Wk_list, W[:, :, k]')
    end                             
end


"""
--------
W update
--------
"""


function _update_W!(W, H_unfold, H_norms, resids, l1_W, l2_W)
    L, N, K = size(W)
    for k = 1:K
        for l = 0:(L-1)
            _update_W_col!(k, l, W, H_unfold, H_norms, resids, l1_W, l2_W)
        end
    end
end


function _update_W_col!(k, l, W, H_unfold, H_norms, resids, l1_W, l2_W)
    L, N, K = size(W)
    ind = l*K + k

    resids .-= W[l+1, :, k] * H_unfold[ind, :]'  # outer product
    W[l+1, :, k] = _next_W_col(H_unfold[ind, :], H_norms[ind], resids, l1_W, l2_W)
    resids .+= W[l+1, :, k] * H_unfold[ind, :]'  # outer product
end

                             
function _next_W_col(Hkl, norm_Hkl, resid, l1_W, l2_W)
    return max.((-resid * Hkl .- l1_W) ./ (norm_Hkl^2 + EPSILON + l2_W), 0.0)
end


"""
--------
H update
--------
"""

function _update_H_regular!(W, H, resids, Wk_list, W_norms, l1_H, l2_H)
    K, T = size(H)
    
    for k = 1:K
        for t = 1:T
            _update_H_entry!(W, H, resids, k, t, Wk_list[k], W_norms, l1_H, l2_H)
        end
    end
end


function _update_H_entry!(W, H, resids, k, t, Wk, W_norms, l1_H, l2_H)
    L, N, K =size(W)
    T = size(H)[2]

    # Compute norm
    norm_Wkt = norm(W_norms[k, 1:min(T-t+1, L)])

    # Remove factor from residual
    remainder = resids[:, t:min(t+L-1, T)] - (H[k, t] * Wk[:, 1:min(T-t+1, L)])

    # Update
    H[k, t] = _next_H_entry(Wk[:, 1:min(T-t+1, L)], norm_Wkt, remainder, l1_H, l2_H)

    # Add factor back to residual
    resids[:, t:min(t+L-1, T)] = remainder + (H[k, t] * Wk[:, 1:min(T-t+1, L)])
end


function _next_H_entry(Wkt, norm_Wkt, remainder, l1_H, l2_H)
    trace = reshape(Wkt, length(Wkt))' * reshape(-remainder, length(remainder))
    return max((trace - l1_H) / (norm_Wkt ^ 2 + EPSILON + l2_H), 0)
end


"""
Expand the factor tensor into a matrix
"""
function _unfold_factor(factor_tens, n_batch, L, N)
    return transpose(reshape(factor_tens, L*n_batch, N))
end


"""
Fold factor into a tensor.
"""
function _fold_factor(Wk, batch)
    return batch * Wk'  # outer product
end


"""
Select the appropriate part of the residual matrix and fold into a tensor.
"""
function _fold_resids(start, n_batch, resids, L, N)
    cropped = resids[:, start:(start + L*n_batch)]
    return reshape(transpose(cropped), n_batch, L*N)
end
