"""
Fit CNMF using Alternating Non-Negative Least Squares.
"""

mutable struct ANLSUpdate <: AbstractCFUpdate
    resids
    data_norm
end

function ANLSUpdate(data, W, H)
    resids = compute_resids(data, W, H)
    data_norm = norm(data)
    return ANLSUpdate(resids, data_norm)
end


# Set tolerance for NNLS solves
const NNLS_TOL = 1e-5


"""Main update rules"""
function update_motifs!(rule::ANLSUpdate, data, W, H; kwargs...)
    _anls_update_W!(data, W, H)
end

function update_feature_maps!(rule::ANLSUpdate, data, W, H; variant=:basic, kwargs...)
    rule.resids = compute_resids(data, W, H)
    
    if variant == :block
        _anls_block_update_H!(rule, W, H)
    else
        _anls_update_H!(rule, W, H)
    end
    
    return norm(rule.resids) / rule.data_norm
end


"""
Private
"""


"""
This is just a single NNLS solve using the unfolded H matrix.
"""
function _anls_update_W!(data, W, H)
    L,N,K = size(W)
    H_unfold = shift_and_stack(H, L)

    W_unfold = nonneg_lsq(
        t(H_unfold), t(data),
        alg=:pivot, variant=:comb, tol=NNLS_TOL
    )
    
    W[:,:,:] = fold_W(t(W_unfold), L, N, K)
end


"""
Perform H update a single column at a time
"""
function _anls_update_H!(rule::ANLSUpdate, W, H; cols=nothing)
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
            rule.resids[:, t:last] -= H[k, t] * W[1:block_size, :, k]'
        end

        unfolded_W = _unfold_W(W)[1:block_size*N, :]
        b = vec(rule.resids[:, t:last])
        
        # Update one column of H
        H[:,t] = nonneg_lsq(unfolded_W, -b,
                            alg=:pivot, variant=:cache,
                            tol=NNLS_TOL)
        
        # Update residual
        for k = 1:K
            rule.resids[:, t:last] += H[k, t] * W[1:block_size, :, k]'
        end
    end
end



"""
Update several columns of H at once.
"""
function _anls_block_update_H!(rule::ANLSUpdate, W, H)
    K, T = size(H)
    L, N, K = size(W)

    for l = 1:L
        inds = l:L:T-L+1
        
        # Remove contribution to residual
        for k = 1:K
            for t in inds
                rule.resids[:, t:t+L-1] -= H[k, t] * W[:, :, k]'
            end
        end
        
        unfolded_W = _unfold_W(W)

        B = zeros(N*L, length(inds))
        for i in 1:length(inds)
            t = inds[i]
            B[:, i] = vec(rule.resids[:, t:t+L-1])
        end

        # Update block of H
        H[:, inds] = nonneg_lsq(unfolded_W, -B,
                                alg=:pivot, variant=:comb,
                                tol=NNLS_TOL)

        # Update residual
        for k = 1:K
            for t in inds
                rule.resids[:, t:t+L-1] += H[k, t] * W[:, :, k]'
            end
        end
    end

    _anls_update_H!(rule, W, H; cols=T-L+2:T)
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
