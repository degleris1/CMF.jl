"""
Fit CMF using alternating optimization, where each update is solved
using ADMM.
"""

mutable struct ADMMUpdate <: AbstractCFUpdate
    datanorm
    U
    Q
end


function ADMMUpdate(data::Matrix, W::Tensor, H::Matrix)
    L, N, K = size(W)
    T = size(H, 2)
    return ADMMUpdate(
        norm(data),
        [zeros(T, N), zeros(L*K, N), zeros(L*K, N)],
        [zeros(N, T), zeros(K, T), zeros(K, T)],
    )
end


function update_motifs!(
    rule::ADMMUpdate, data, W, H; rhow=1, admm_W_maxiter=15, kwargs...
)
    L, N, K = size(W)
    T = size(H, 2)

    # Precompute reused quantities
    Hstk = shift_and_stack(H, L)

    HstkT = copy(Hstk')
    HHTI = (Hstk * Hstk') + (2*I)(L*K)
    HHTI_fact = cholesky(HHTI)


    # Initialize auxilliary and dual variables
    loss_hist = []

    Wstk = zeros(L*K, N)
    Z1 = zeros(T, N)
    Z2 = zeros(L*K, N)
    Z3 = zeros(L*K, N)

    U1 = rule.U[1]
    U2 = rule.U[2]
    U3 = rule.U[3]

    estT = zeros(T, N)
    dataT = copy(data')
    V2 = zeros(size(U2))
    rhs = zeros(K*L, N)
    weights = zeros(1, N)

    for iter = 1:admm_W_maxiter
        # Update W itself
        mul!(rhs, Hstk, Z1-U1)
        ldiv!(Wstk, HHTI_fact, rhs + Z2-U2 + Z3-U3)
    
        # Update wrt primary objective
        mul!(estT, HstkT, Wstk)
        @. Z1 = (1 / (1 + 1/rhow)) * ((estT + U1) + (1/rhow) * dataT)

        # Update wrt norm constraint
        @. V2 = Wstk + U2
        sum!(weights, V2 .^ 2)
        for n = 1:N
            if weights[n] >= 1
                @. Z2[:, n] = V2[:, n] / sqrt(weights[n])
            end
        end

        # Update wrt nonnegativity
        @. Z3 = max(0, Wstk+U3)

        # Update dual variable
        @. U1 += estT - Z1
        @. U2 += Wstk - Z2
        @. U3 += Wstk - Z3
        
        #push!(loss_hist, norm(data - Wstk'Hstk) / rule.datanorm)
    end

    #plt.plot(loss_hist, ls="--")

    W .= fold_W(Z3', L, N, K)
end


function update_feature_maps!(
    rule::ADMMUpdate, data, W, H; 
    rhoh=4, admm_H_maxiter=15, l1H=0, kwargs...
)

    # Update feature maps H to solve
    # minimize \|Z_1 - B\|_2^2 + \|Z_2\|_1 + I_+(Z_3)
    # subject to        Z1 = \sum_k W_k * h_k
    #                   Z2 = H
    #                   Z3 = H

    # Step 1) Update primal variable
    #               nH = arg min \| \sum_k Wk * hk - Z1 + Q1 \|_2^2 
    #                               + \| H - Z2 + Q2 \|_2^2 
    #                               + \| H - Z3 + Q3 \|_2^2
    # Step 2) Update auxiliary variables
    #               Z1 = arg min D(Z1 - B)+ (1/rho) \|Z1 - est - Q1\|_2  
    #                       (least squares when D(X) = \| X \|_2^2)
    #               Z2 = arg min \| Z2 \|_1 + (1/rho) \| Z2 - H - Q2 \|_2        
    #                       (shrinkage)
    #               Z3 = arg min I_+(Z3) + (1\rho) \|Z3 - H - Q3 \|_2       
    #                       (projection)
    # Step 3) Update dual variables

    #Initialize auxilliary and dual variables
    loss_hist = []

    Z1 = zeros(size(data))  # Z1 = tensor_conv(W, H), handles loss function
    Z2 = zeros(size(H))  # Z2 = H, handles
    Z3 = zeros(size(H))  # Z3 = H

    Q1 = rule.Q[1]  # dual of Z1's equality constraint
    Q2 = rule.Q[2]  # dual of Z2's equality constraint
    Q3 = rule.Q[3]  # dual of Z3's equality constraint
    Q1 .= 0
    Q2 .= 0
    Q3 .= 0

    normW = norm(W)
    L, N, K = size(W)
    T = size(H, 2)

    Wpad = zeros(T, N, K)
    Wpad[1:L, :, :] = W

    whc, gram_arr = precompute_solveH(Wpad, 1)

    for iter = 1:admm_H_maxiter
        # Update primal variable H
        fastsolveH!(H, gram_arr, whc, Z1-Q1, Z2-Q2, Z3-Q3, 1)

        # Update Z1 (diagonal quadratic)
        # TODO this should be circular

        est = tensor_conv(W, H)  # <-- slow line, 10 % of time here
        @. Z1 = (1 / (1 + 1/rhoh)) * ((est + Q1) + (1/rhoh) * data)

        # Update Z2 (shrinkage)
        @. Z2 = sign(H + Q2) * max(0, abs(H + Q2) - l1H/rhoh)

        # Update Z3 (projection)
        @. Z3 = max(0, H + Q3)

        # Update dual variable Q
        @. Q1 += est - Z1
        @. Q2 += H - Z2
        @. Q3 += H - Z3

        # push!(
        #     loss_hist, 
        #     (norm(data - est)^2 + sum(Z2) 
        #     + norm(Z2 - H)^2 + norm(Z3 - H)^2 + norm(Z1 - est)^2)
        # )
    end

    #@show norm(Z3 - H)^2 / (K*T)
    #@show minimum(H)
    #@show sum(H .< 0)
    H .= Z3

    #plt.plot(loss_hist, ls="--")

    return norm(compute_resids(data, W, H)) / rule.datanorm
end


function precompute_solveH(W, rho)
    T, N, K = size(W)
    
    # Permute wh to be memory efficient -- K N T
    wh = permutedims(fft(W, 1), (3, 2, 1))
    whc = conj.(wh)

    gram_arr = []
    for t = 1:T
        push!(gram_arr, cholesky(whc[:, :, t] * whc[:, :, t]' + 2*rho*I(K)))
    end

    return whc, gram_arr
end

function fastsolveH!(H, gram_arr, whc, B, V1, V2, rho)
    K, N, T = size(whc)

    bh = fft(B, 2)
    v1h = fft(V1, 2)
    v2h = fft(V2, 2)

    hh = zeros(ComplexF64, K, T)
    hh_ws = zeros(ComplexF64, K)
    beta = zeros(ComplexF64, K)
    @inbounds for t = 1:T
        @views mul!(beta, whc[:, :, t], bh[:, t])
        @. beta += sqrt(rho) * (v1h[:, t] + v2h[:, t])
        ldiv!(hh_ws, gram_arr[t], beta)
        @views @. hh[:, t] = hh_ws  # TODO can this line just fit into ldiv?
    end

    H .= real.(ifft(hh, 2))
end

function solveH(W, B, V1, V2, rho)
    T, N, K = size(W)

    wh = fft(W, 1)
    whc = conj.(wh)
    bh = fft(B, 2)
    v1h = fft(V1, 2)
    v2h = fft(V2, 2)

    hh = zeros(Complex{Float64}, K, T)
    for t = 1:T
        A = zeros(Complex{Float64}, N+2K, K)
        beta = zeros(Complex{Float64}, N+2K)
        for n = 1:N
            A[n, :] = whc[t, n, :]'
            beta[n] = bh[n, t]
        end
        A[end-2K+1 : end-K, :] = sqrt(rho) * I(K)
        A[end-K+1 : end, :] = sqrt(rho) * I(K)
        beta[end-2K+1: end-K, :] = sqrt(rho) * v1h[:, t]
        beta[end-K+1:end, :] = sqrt(rho) * v2h[:, t]

        hh[:, t] = A \ beta
    end
    return real.(ifft(hh, 2))
end


#=
    R = conv(W, H)
    Λ = exp(R)
    y_{ij} \sim Poisson(Λ_{ij})

    J(Λ) = Σ_{ij} -y_{ij} R_{ij} + exp(R_{ij}) + w (R_{ij} - E_{ij})^2

    ∂_{ij}J = -y_{ij} + exp(R_{ij}) + 2 w (R_{ij} - E_{ij})
    0 = ...
    y_{ij} + 2 w E_{ij} = exp(R_{ij}) + 2 w R_{ij}
=#