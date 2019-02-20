module ANNLS
"""
Fit CNMF using Alternating Non-Negative Least Squares.
Note: this requires having the NonNegLeastSquares source available 
in a directory adjacent to the cmf.jl directory.
"""

# Import 
push!(LOAD_PATH, "../../")
import NonNegLeastSquares
using LinearAlgebra
include("./common.jl")

mutable struct ANNLSmeta
    resids
    data_norm
end

"""
Main update rule
"""
function update(data, W, H, meta, options)
    if (meta == nothing)
        meta = _initialize_meta(data, W, H)
    end

    # W update
    _update_W!(data, W, H)

    if get(options, "mode", nothing) == "parallel"
        _update_H_parallel!(data, W, H)
    else
        _update_H!(data, W, H)
    end

    meta.resids = tensor_conv(W, H) - data
    return norm(meta.resids) / meta.data_norm, meta
end

function _initialize_meta(data, W, H)    
    resids = tensor_conv(W, H) - data
    data_norm = norm(data)
    return ANNLSmeta(resids, data_norm)
end

function _update_W!(data, W, H)
    """
    This is just a single NNLS solve using the unfolded H
    matrix.
    """
    L,N,K = size(W)
    H_unfold = shift_and_stack(H, L)
    W_unfold = NonNegLeastSquares.nonneg_lsq(t(H_unfold), t(data), alg=:pivot)
    W[:,:,:] = fold_W(t(W_unfold), L, N, K)
end

"""
Perform H update updating every L colums of H simultaneously.
"""
function _update_H_parallel!(data, W, H)
    K, T = size(H)
    L, N, K = size(W)

    # Update every L columns of H
    ind_set_total = []
    for start in 1:L
        ind_set = start:L:T
        B = zeros(N,length(ind_set))
        for (i, t) in enumerate(ind_set)
            B[:,i] = form_b_vec(data, W, H, t)
        end
        # Solve L NNLS problems
        H[:,ind_set] = NonNegLeastSquares.nonneg_lsq(W[1,:,:], B, alg=:pivot)
    end
end

"""
Perform H update a single column at a time
"""
function _update_H!(data, W, H)
    K, T = size(H)
    L, N, K = size(W)

    for t in 1:T
        b = form_b_vec(data, W, H, t)
        H[:,t] = NonNegLeastSquares.nonneg_lsq(W[1,:,:], b, alg=:pivot)
    end
end

"""
Forms the vector b for the nonnegative least squres objective
||Ax - b|| required to update column t of H
"""
function form_b_vec(data, W, H, t)
    L, N, K = size(W)
    v = zeros(N,1)
    for l = 1:L-1
        h_idx = t-l
        if h_idx <= 0
            continue
        end
        v += W[l+1,:,:] * H[:,h_idx]
    end
    return data[:,t] - v
end

"""
Fold W_tilde (a block matrix) into a W tensor
"""
function fold_W(W_mat, L, N, K)
    W_tens = zeros(L, N, K)
    for l in 0:L-1
        W_fac = W_mat[:,1+(K*l):K*(l+1)]
        W_tens[l+1,:,:] = W_fac
    end
    return W_tens
end

"""
Convenience function for transpose
"""
function t(A)
    return permutedims(A, (2,1))
end

end  # module
