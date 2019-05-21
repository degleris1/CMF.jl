""" Generate separable data. """
function gen_sep_data(N, T, K, L; H_sparsity=0.75)

    # Generate W
    W = 0.5 .+ rand(L, N, K) 
    
    # Generate H
    H = rand(K, T) .* (rand(K, T) .> H_sparsity)

    # Add separable factors
    if (T < 3*K*L)
        println("T too small")
        return nothing
    end

    hL = floor(Int, L/2)
    times = collect(1:T-L)
    free = (times .!= 0)
    
    for k = 1:K
        for (down, up) in [(-L, hL), (-hL, L)]  # Left and ride side of sequence 
            t = rand(times[free])
            t1 = max(1, t+down)
            t2 = min(T, t+up)

            H[:, t1:t2] .= 0
            H[k, t] = 0.5 + rand()

            free[t1:min(t2,T-L)] .= false
        end
    end
    
    # Generate data
    X = tensor_conv(W, H)
    
    return X, W, H
end
