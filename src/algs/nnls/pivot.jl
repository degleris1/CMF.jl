"""
Subroutines for solving the NNLS subproblems for CNMF using the Pivot methods
implemented by NonNegLeastSquares.jl.
"""

using NonNegLeastSquares


"""
Pivot update for W. Creates large block matrix H_tilde and solves for entire W
matrix at once.
"""
function pivot_update_W!(data, W, H)
    L,N,K = size(W)
    H_unfold = shift_and_stack(H, L)
    W_unfold = nonneg_lsq(t(H_unfold), t(data), alg=:pivot, variant=:comb)
    W[:,:,:] = fold_W(t(W_unfold), L, N, K)
end

"""
Perform H update a single column at a time
"""
function pivot_update_H_cols!(W, H, meta; cols=nothing)
    N, T, K, L = unpack_dims(W, H)
    if cols === nothing
        cols = 1:T
    end
    
    for t in cols
        last = min(t+L-1, T)
        block_size = last - t + 1
        
        # Remove contribution to residual
        for k = 1:K
            meta.resids[:, t:last] -= H[k, t] * W[1:block_size, :, k]'
        end

        unfolded_W = _unfold_W(W)[1:block_size*N, :]
        b = vec(meta.resids[:, t:last])
        
        # Update one column of H
        H[:,t] = nonneg_lsq(unfolded_W, -b, alg=:pivot, variant=:cache)
        
        # Update residual
        for k = 1:K
            meta.resids[:, t:last] += H[k, t] * W[1:block_size, :, k]'
        end
    end
end

"""
Update several columns of H at once using Pivot method
"""
function pivot_block_update_H!(W, H, meta)
    K, T = size(H)
    L, N, K = size(W)

    for l = 1:L
        inds = l:L:T-L+1
        
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
        H[:, inds] = NonNegLeastSquares.nonneg_lsq(unfolded_W, -B,
                                                   alg=:pivot, variant=:comb)
        
        # Update residual
        for k = 1:K
            for t in inds
                meta.resids[:, t:t+L-1] += H[k, t] * W[:, :, k]'
            end
        end
    end

    pivot_update_H_cols!(W, H, meta; cols=T-L+2:T)
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


function _unfold_W(W)
    L, N, K = size(W)
    return reshape(permutedims(W, (2,1,3)), N*L, K)
end
