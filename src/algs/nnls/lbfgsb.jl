"""
Subroutines for solving the NNLS subproblems for CNMF using the LBFGS-B
algorithm for box-constrained optimization.

Uses the Julia package LBFGSB which is a wrapper for a FORTRAN implementation.
See https://github.com/Gnimuc/LBFGSB.jl for details.
"""

using LBFGSB
using LinearAlgebra

function LBFGSB_update_H!(W, H, meta; warm_start=true)
    N, T, K, L = unpack_dims(W, H)

    for t in 1:T
        last = min(t+L-1, T)
        block_size = last - t + 1
        
        # Remove contribution to residual
        for k = 1:K
            meta.resids[:, t:last] -= H[k, t] * W[1:block_size, :, k]'
        end

        unfolded_W = _unfold_W(W)[1:block_size*N, :]
        b = vec(meta.resids[:, t:last])
        
        # Update one column of H
        Ax(x) = unfolded_W*x
        ATx(x) = unfolded_W'*x

        h_col_guess = nothing
        if warm_start
            h_col_guess = H[:,t]
        end
        H[:,t] = LBFGS_B_solve(Ax, ATx, -b, K; X_init=h_col_guess)
        
        # Update residual
        for k = 1:K
            meta.resids[:, t:last] += H[k, t] * W[1:block_size, :, k]'
        end
    end
end

function LBFGSB_update_W!(data, W, H; warm_start=true)
    N, T, K, L = unpack_dims(W, H)
    unfolded_W = w_tilde(W)
    println(N, T, L, K)
    
    # solve using LBFGSB one row of W_tilde at a time
    Ax(x) = H_tilde_transpose_x(H, x, L, K, T)
    ATx(x) = H_tilde_x(H, x, L, K, T)
    for i = 1:N

        w_row_guess = nothing
        if warm_start
            w_row_guess = unfolded_W[i,:]
        end
        unfolded_W[i,:] = LBFGS_B_solve(Ax, ATx, data[i,:], K*L; X_init=w_row_guess)
    end
    W[:,:,:] = fold_W(unfolded_W, L, N, K)
end

function LBFGS_B_solve(Ax::Function,
                       ATx::Function,
                       Y::Vector{Float64},
                       xdim::Integer;
                       X_init = nothing,
                       memcor=15
                       )

    f(X) = 0.5 * norm(Ax(X) - Y)^2
    g(X) = ATx(Ax(X) - Y)
    func(X) = f(X), g(X)

    if X_init === nothing
        X_init = zeros(xdim)
    end

    optimizer = L_BFGS_B(xdim, memcor)

    # bounds -- only lower bound
    bounds = zeros(3,xdim)
    for i = 1:xdim
    bounds[1,i] = 1
    bounds[2,i] = 0.0
    end

    fout, xout = optimizer(func, X_init, bounds)
    return xout
end

"""
function H_tilde_transpose_x(H, x, L, K, T)
"""
function H_tilde_transpose_x(H, x, L, K, T)
    out = zeros(T)
    for l = 1:L
        h_c = view(H', 1:T-l+1,:)
        x_c = view(x, (l-1)*K+1:l*K)
        out_c = view(out, l:T)
        out_c[:] += h_c*x_c
    end 
    return out
end

"""
function H_tilde_x(H, x, L, K, T)
"""
function H_tilde_x(H, x, L, K, T)
    out = zeros(L*K)
    for l = 1:L
        h_c = view(H, :, 1:T-l+1)
        x_c = view(x, l:T)
        out_c = view(out, (l-1)*K+1:l*K)
        out_c[:] = h_c*x_c
    end
    return out 
end