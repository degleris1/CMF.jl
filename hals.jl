module HALS


# Imports
using LinearAlgebra
include("./common.jl")


"""
Main update rule
"""
function update(data, W, H, meta, options)
    if (meta == nothing)
        meta = _initialize_meta!(data, W, H)
    end

    # W update
    _setup_W_update!(W, H, meta)
    _update_W!(W, meta.H_unfold, meta.H_norms, meta.resids)

    # H update
    _setup_H_update!(W, H, meta)
    if get(options, "mode", nothing) == "elementwise"
        _update_H_simple!(W, H, meta.resids, meta.W_norms)
    else  # regular
        _update_H!(W, H, meta.resids, meta.W_norms)
    end
    
    return norm(meta.resids) / meta.data_norm, meta
end



"""
----------------------------
Internals and Initialization
----------------------------
"""


mutable struct HALSMeta
    resids  # Internals
    data_norm
    batch_inds
    batch_sizes
    
    H_unfold  # W setup
    H_norms

    W_norms  # H setup
    W_raveled
    W_clones
end


function _initialize_meta!(data, W, H)
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
        for l = 1:L
            batch = range(l, stop=T-L, step=L)
            push!(batch_inds[k], batch)
            push!(batch_sizes[k], length(batch))
        end
    end
    
    return HALSMeta(resids, data_norm, batch_inds, batch_sizes,  # Internals
                    nothing, nothing,  # W setup
                    nothing, nothing, nothing)  # H setup
end



"""
--------
W update
--------
"""


function _update_W!(W, H_unfold, H_norms, resids)
    L, N, K = size(W)
    for k = 1:K
        for l = 0:(L-1)
            _update_W_col!(k, l, W, H_unfold, H_norms, resids)
        end
    end
end


function _update_W_col!(k, l, W, H_unfold, H_norms, resids)
    L, N, K = size(W)
    ind = l*K + k

    resids .-= W[l+1, :, k] * H_unfold[ind, :]'  # outer product
    W[l+1, :, k] = _next_W_col(H_unfold[ind, :], H_norms[ind], resids)
    resids .+= W[l+1, :, k] * H_unfold[ind, :]'  # outer product
end

                             
function _next_W_col(Hkl, norm_Hkl, resid)
    return max.((-resid * Hkl) ./ (norm_Hkl^2 + EPSILON), 0.0)
end


function _setup_W_update!(W, H, meta)
    L, N, K = size(W)

    meta.H_unfold = shift_and_stack(H, L)  # Unfold matrices
    meta.H_norms = zeros(K*L)  # Compute norms
    for i=1:(K*L)
        meta.H_norms[i] = norm(meta.H_unfold[i, :])
    end
end


"""
--------
H update
--------
"""

function _update_H!(W, H, resids, W_norms)
    println("Not yet implemented!")
end


function _update_H_simple!(W, H, resids, W_norms)
    K, T = size(H)
    
    for k = 1:K
        for t = 1:T
            _update_H_entry!(W, H, resids, k, t, W_norms)
        end
    end
end


function _update_H_entry!(W, H, resids, k, t, W_norms)
    L, N, K =size(W)
    T = size(H)[2]

    # Collect cached data
    Wk = W[:, :, k]'
    norm_Wkt = norm(W_norms[k, 1:min(T-t+1, L)])

    # Remove factor from residual
    remainder = resids[:, t:min(t+L-1, T)] - (H[k, t] * Wk[:, 1:min(T-t+1, L)])

    # Update
    H[k, t] = _next_H_entry(Wk[:, 1:min(T-t+1, L)], norm_Wkt, remainder)

    # Add factor back to residual
    resids[:, t:min(t+L-1, T)] = remainder + (H[k, t] * Wk[:, 1:min(T-t+1, L)])
end


function _next_H_entry(Wkt, norm_Wkt, remainder)
    trace = reshape(Wkt, length(Wkt))' * reshape(-remainder, length(remainder))
    return max(trace / (norm_Wkt ^ 2 + EPSILON), 0)
end


function _update_H_batch!()
    return nothing
end


function _next_H_batch(Wk, norm_Wk, remainder)
    traces = 0
end


function _setup_H_update!(W, H, meta)
    L, N, K = size(W)

    # Setup norms
    meta.W_norms = zeros(K, L)
    for k = 1:K
        for l = 1:L
            meta.W_norms[k, l] = norm(W[l, :, k])
        end
    end

    # Setup raveled matrices and clones
    W_raveled = []
    for k = 1:K
        push!(W_raveled, reshape(W[:, :, k], N*L))
    end

    meta.W_raveled = W_raveled
end


"""
Expand the factor tensor into a matrix
"""
function _unfold_factor(factor_tens, n_batch)
    return transpose(reshape(factor_tens, L*n_batch, n))
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
