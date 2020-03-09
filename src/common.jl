"""
Abstract type for an algorithm that fits a convolutional factorization.

Must implement:
    fit(
        alg::AlternatingOptimizer,
        data::Matrix,
        L::Int64,
        K::Int64,
        W_init::Tensor,
        H_init::Matrix;
        kwargs...
    )
"""
abstract type AbstractCFAlgorithm end

function tensor_conv(W::Tensor, H::Matrix)
    L, N, K = size(W)
    T = size(H)[2]

    pred = zeros(N, T)
    for lag = 0:(L-1)
        pred[:, lag+1:T] .+= s_dot(W[lag+1, :, :], H, lag)
    end
    return pred

    # L, N, K = size(W)
    # T = size(H, 2)
    
    # return copy(reshape(permutedims(W, [2, 3, 1]), N, L*K)) * shift_and_stack(H, L)
end

"""Computes normalized quadratic loss."""
compute_loss(data::Matrix, W::Tensor, H::Matrix) =
    norm(compute_resids(data, W, H)) / norm(data)

"""Computes matrix of residuals."""
compute_resids(data::Matrix, W::Tensor, H::Matrix) = 
    tensor_conv(W, H) - data


function tensor_transconv(W::Tensor, X::Matrix)
    L, N, K = size(W)
    T = size(X)[2]

    result = zeros(K, T)
    for lag = 0:(L-1)
        result[:, 1:T-lag] += W[lag+1, :, :]' * shift_cols(X, -lag)
    end

    return result
end


function s_dot(Wl::Matrix, H::Matrix, lag)
    K, T = size(H)

    if (lag < 0)
        return Wl * H[:, 1-lag:T]

    else  # lag >= 0
        return Wl * H[:, 1:T-lag]
    end
end


function shift_cols(X::Matrix, lag)
    T = size(X)[2]
    
    if (lag <= 0)
        return X[:, 1-lag:T]

    else  # lag > 0
        return X[:, 1:T-lag]
    end
end


function shift_and_stack(H::Matrix, L)
    K, T = size(H)

    H_stacked = zeros(L*K, T)
    for lag = 0:(L-1)
        H_stacked[1+K*lag:K*(lag+1), 1+lag:T] = shift_cols(H, lag)
    end

    return H_stacked
end


function unpack_dims(W::Tensor, H::Matrix)
    L, N, K = size(W)
    T = size(H, 2)

    return N, T, K, L
end
