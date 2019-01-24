
function gen_synthetic(;K=3, N=100, L=20, T=50, H_sparsity=0.9, noise_scale=1.0)
    # Generate random convolutional parameters
    W = zeros(L, N, K)
    H = rand(K, T)

    # Add sparsity
    H = H .* (rand(K, T) .> H_sparsity) 
    
    # Add structure
    for i = 1:N
        j = rand(1:K)
        W[:, i, j] += _gauss_plus_delay(L)
    end

    # Add noise
    noise = noise_scale * rand(N, T)
    data = tensor_conv(W, H) + noise

    return data
end


function _gauss_plus_delay(n_steps)
    tau = rand() * 3 - 1.5
    x = range(-3-tau, stop=3-tau, length=n_steps)
    y = exp.(-x.^2)
    return y / maximum(y)
end
;
