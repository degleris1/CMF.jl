module HALS

include("./common.jl")


"""
Main update rule
"""
function update!(data, W, H, meta; l1_H=0, l2_H=0,
                 l1_W=0, l2_W=0, kwargs...)
   
    if (meta == nothing)
        meta = HALSMeta(data, W, H)
    end

    # W update
    _setup_W_update!(W, H, meta)
    _update_W!(W, meta.H_unfold, meta.H_norms, meta.resids, l1_W, l2_W)

    # H update
    _setup_H_update!(W, H, meta)
    if get(kwargs, :parallel, false)
        _update_H_parallel!(W, H, meta.resids,
                   meta.batch_inds, meta.batch_sizes,
                   meta.Wk_list, meta.W_norms,
                   l1_H, l2_H)
    else  # regular
        _update_H_regular!(W, H, meta.resids, meta.Wk_list, meta.W_norms,
                           l1_H, l2_H)
    end
    
    return norm(meta.resids) / meta.data_norm, meta
end



"""
-----------------------------------
Internals, Initialization and Setup
-----------------------------------
"""


mutable struct HALSMeta
    resids  # Internals
    data_norm
    batch_inds
    batch_sizes
    
    H_unfold  # W setup
    H_norms

    W_norms  # H setup
    Wk_list

    function HALSMeta(data, W, H)
        L, N, K = size(W)
        T = size(H)[2]
    
        resids = tensor_conv(W, H) - data
        data_norm = norm(data)

        # Set up batches
        batch_inds = []
        batch_sizes = []
        for k = 1:K
            push!(batch_sizes, [])
            push!(batch_inds, [])
            for l = 0:(L-1)
                batch = (l+1):L:(T-L)
                push!(batch_inds[k], batch)
                push!(batch_sizes[k], length(batch))
            end
        end
        
        return new(resids, data_norm, batch_inds, batch_sizes,  # Internals
                   nothing, nothing,  # W setup
                   nothing, nothing)  # H setup
    end
end


function _setup_W_update!(W, H, meta)
    L, N, K = size(W)

    meta.H_unfold = shift_and_stack(H, L)  # Unfold matrices
    meta.H_norms = zeros(K*L)  # Compute norms
    for i=1:(K*L)
        meta.H_norms[i] = norm(meta.H_unfold[i, :])
    end
end


function _setup_H_update!(W, H, meta)
    L, N, K = size(W)
    
    # Set up norms
    meta.W_norms = zeros(K, L)
    for k = 1:K
        for l = 1:L
            meta.W_norms[k, l] = norm(W[l, :, k])
        end
    end
                                    
    # Slice W by k
    meta.Wk_list = []
    for k = 1:K
        push!(meta.Wk_list, W[:, :, k]')
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

function _update_H_parallel!(W, H, resids, batch_inds, 
                             batch_sizes, Wk_list, W_norms,
                             l1_H, l2_H)
    L, N, K = size(W)
    T = size(H)[2]

    for k = 1:K
        norm_Wk = norm(W_norms[k, :])
        for l = 0:(L-1)
            _update_H_batch!(W, H, resids, k, l,
                             batch_inds[k][l+1], batch_sizes[k][l+1],
                             Wk_list[k], norm_Wk,
                             l1_H, l2_H)
            _update_H_entry!(W, H, resids, k, T-L+l+1, Wk_list[k], W_norms,
                             l1_H, l2_H)
        end
    end
end


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


function _update_H_batch!(W, H, resids, k, l, batch_ind, n_batch, Wk, norm_Wk, l1_H, l2_H)
    L, N, K = size(W)

    # Set up batch
    batch = H[k, batch_ind]
    end_batch = l + L*n_batch
    
    # Remove factor from residual
    _add_factor!(resids, batch, -Wk, batch_ind, n_batch, L)

    # Update H
    H[k, batch_ind] = _next_H_batch(Wk, norm_Wk, resids, batch_ind, n_batch, L, l1_H, l2_H)

    # Add factor back to residual
    new_batch = H[k, batch_ind]
    _add_factor!(resids, new_batch, Wk, batch_ind, n_batch, L)
end


function _next_H_batch(Wk, norm_Wk, resids, batch_ind, n_batch, L, l1_H, l2_H)
    traces = zeros(n_batch)
    for i = 1:n_batch
        j = batch_ind[i]
        traces[i] = reshape(Wk, length(Wk))' * reshape(resids[:, j:j+L-1], length(Wk))
    end
    return max.( (traces .- l1_H) / (norm_Wk ^ 2 + EPSILON + l2_H), 0)
end


function _add_factor!(resids, batch, Wk, batch_ind, n_batch, L)
    for i = 1:n_batch
        j = batch_ind[i]
        resids[:, j:j+L-1] .+= batch[i] * Wk
    end
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

end  # module
;
