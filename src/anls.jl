module ANLS
"""
Fit CNMF using Alternating Non-Negative Least Squares.
"""

# Import 
using NonNegLeastSquares
using LinearAlgebra
include("./common.jl")


"""
Main update rule
"""
function update!(data, W, H, meta; variant=:cache, block=false, kwargs...)
    if (meta == nothing)
        meta = ANLSmeta(data, W, H)
    end

    # W update
    _update_W!(data, W, H, variant=variant)
    meta.resids = compute_resids(data, W, H)

    # H update
    if (block)
        _block_update_H!(W, H, meta, variant=variant)
    else
        _update_H!(W, H, meta, variant=variant)
    end
    
    return norm(meta.resids) / meta.data_norm, meta
end


"""
Private
"""


mutable struct ANLSmeta
    resids
    data_norm
    function ANLSmeta(data, W, H)
        resids = compute_resids(data, W, H)
        data_norm = norm(data)
        return new(resids, data_norm)
    end
end


"""
This is just a single NNLS solve using the unfolded H matrix.
"""
function _update_W!(data, W, H; variant=:cache)
    L,N,K = size(W)
    H_unfold = shift_and_stack(H, L)

    if (variant == nothing)
        W_unfold = nonneg_lsq(t(H_unfold), t(data), alg=:pivot)
    else
        W_unfold = nonneg_lsq(t(H_unfold), t(data), alg=:pivot, variant=variant)
    end
    
    W[:,:,:] = fold_W(t(W_unfold), L, N, K)
end


"""
Perform H update a single column at a time
"""
function _update_H!(W, H, meta; variant=:cache, cols=nothing)
    N, T, K, L = unpack_dims(W, H)

    if (cols == nothing)
        inds = 1:T
    else
        inds = cols
    end
    
    for t in inds  
        last = min(t+L-1, T)
        block_size = last - t + 1
        
        # Remove contribution to residual
        for k = 1:K
            meta.resids[:, t:last] -= H[k, t] * W[1:block_size, :, k]'
        end

        unfolded_W = _unfold_W(W)[1:block_size*N, :]
        b = vec(meta.resids[:, t:last])
        
        # Update one column of H
        if (variant == nothing)
            H[:,t] = nonneg_lsq(unfolded_W, -b, alg=:pivot)
        else
            H[:,t] = nonneg_lsq(unfolded_W, -b, alg=:pivot, variant=variant)
        end
        
        # Update residual
        for k = 1:K
            meta.resids[:, t:last] += H[k, t] * W[1:block_size, :, k]'
        end
    end
end



"""
Update several columns of H at once.
"""
function _block_update_H!(W, H, meta; variant=variant)
    K, T = size(H)
    L, N, K = size(W)

    for l = 1:L
        inds = 1:L:T-L+1
        
        # Remove contribution to residual
        for k = 1:K
            for t in inds
                meta.resids[:, t:t+L-1] -= H[k, t] * W[:, :, k]'
            end
        end
        
        unfolded_W = _unfold_W(W)

        B = zeros(N*L, length(inds))
        for i in 1:length(inds)
            t = inds[i]
            B[:, i] = vec(meta.resids[:, t:t+L-1])
        end

        # Update block of H
        if (variant == nothing)
            H[:, inds] = NonNegLeastSquares.nonneg_lsq(unfolded_W, -B,
                                                   alg=:pivot)
        else
            H[:, inds] = NonNegLeastSquares.nonneg_lsq(unfolded_W, -B,
                                                   alg=:pivot, variant=variant)
        end
        
        # Update residual
        for k = 1:K
            for t in inds
                meta.resids[:, t:t+L-1] += H[k, t] * W[:, :, k]'
            end
        end
    end

    _update_H!(W, H, meta; variant=variant, cols=T-L+2:T)
end


function _unfold_W(W)
    L, N, K = size(W)
    return reshape(permutedims(W, (2,1,3)), N*L, K)
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
