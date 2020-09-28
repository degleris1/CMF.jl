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


function tensor_conv(W::AbstractArray, H::AbstractArray)
    K, N, L = size(W)
    est = zeros(N, size(H, 2))
    return tensor_conv!(est, W, H)
end


function tensor_conv!(est, W::AbstractArray, H::AbstractArray)
    K, N, L = size(W)
    T = size(H, 2)

    @. est = 0
    for lag = 0:(L-1)
        @views s_dot!(est[:, lag+1:T], W[:, :, lag+1]', H, lag, 1, 1)
    end
    
    return est
end


function tensor_circconv!(est, whc, H, hh, esth)
    K, N, T = size(whc)

    @. hh = H
    fft!(hh, 2)

    for t = 1:T
        for n = 1:N
            @views esth[n, t] = whc[:, n, t]'hh[:, t] 
        end
    end

    ifft!(esth, 2)
    @. est = real(esth)
end


"""Computes normalized quadratic loss."""
compute_loss(data::Matrix, W::Tensor, H::Matrix) = error("compute_loss has been deprecated.")
#    norm(compute_resids(data, W, H)) / norm(data)


"""Computes matrix of residuals."""
compute_resids(data::Matrix, W::Tensor, H::Matrix) = 
    tensor_conv(W, H) - data


function tensor_transconv(W::Tensor, X::Matrix)
    K, N, L = size(W)
    T = size(X, 2)

    result = zeros(K, T)
    return tensor_transconv!(result, W, X)
end


function tensor_transconv!(out, W::Tensor, X::Matrix)
    K, N, L = size(W)
    T = size(X, 2)

    @. out = 0
    for lag = 0:(L-1)
        @views mul!(out[:, 1:T-lag], W[:, :, lag+1], shift_cols(X, -lag), 1, 1)
    end

    return out
end


function s_dot(Wl::Matrix, H::Matrix, lag)
    T = size(H, 2)
    N = size(Wl)
    out = zeros(N, T)
    return s_dot!(out, Wl, H, lag)
end


"""
returns B = Wl H S_{l}
"""
function s_dot!(B, Wl, H, lag)
    return s_dot!(B, Wl, H, lag, 1, 0)
end


function s_dot!(B, Wl, H, lag, α, β)
    K, T = size(H)
    
    if lag < 0
        @views mul!(B, Wl, H[:, 1+lag:T], α, β)  # TODO check
    else  # lag >= 0
        @views mul!(B, Wl, H[:, 1:T-lag], α, β)
    end

    return B
end


function shift_cols(X::Matrix, lag)
    T = size(X, 2)
    
    if (lag <= 0)
        return view(X, :, 1-lag:T)

    else  # lag > 0
        return view(X, :, 1:T-lag)
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
    K, N, L = size(W)
    T = size(H, 2)

    return N, T, K, L
end
