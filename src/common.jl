using LinearAlgebra

EPSILON = eps()


function compute_loss(data, W, H)
    return norm(compute_resids(data, W, H)) / norm(data)
end


function compute_resids(data, W, H)
    return tensor_conv(W, H) - data
end


function tensor_conv(W, H)
    L, N, K = size(W)
    T = size(H)[2]

    pred = zeros(N, T)
    for lag = 0:(L-1)
        pred[:, lag+1:T] += s_dot(W[lag+1, :, :], H, lag)
    end
    return pred
end


function tensor_transconv(W, X)
    L, N, K = size(W)
    T = size(X)[2]

    result = zeros(K, T)
    for lag = 0:(L-1)
        result[:, 1:T-lag] += W[lag+1, :, :]' * shift_cols(X, -lag)
    end

    return result
end


function s_dot(Wl, H, lag)
    K, T = size(H)

    if (lag < 0)
        return Wl * H[:, 1-lag:T]

    else  # lag >= 0
        return Wl * H[:, 1:T-lag]
    end
end


function shift_cols(X, lag)
    T = size(X)[2]
    
    if (lag <= 0)
        return X[:, 1-lag:T]

    else  # lag > 0
        return X[:, 1:T-lag]
    end
end


function shift_and_stack(H, L)
    K, T = size(H)

    H_stacked = zeros(L*K, T)
    for lag = 0:(L-1)
        H_stacked[1+K*lag:K*(lag+1), 1+lag:T] = shift_cols(H, lag)
    end

    return H_stacked
end

"""
Return a shift matrix of a given size and shift. 
"""
function shift_matrix(size, shift)
    eye = Matrix{Float64}(I, size, size)
    if shift >= 0
        eye = eye[:,1:end-shift]
        shift = [zeros(size, shift) eye]
        return shift
    else
        shift = abs(shift)
        eye = eye[:,1+shift:end]
        shift = [eye zeros(size, shift)]
        return shift
    end
end

;
