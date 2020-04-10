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
    K, N, L = size(W)
    T = size(H, 2)
    return ADMMUpdate(
        norm(data),
        [rand(T, N), rand(L*K, N), rand(L*K, N)],
        [zeros(N, T), zeros(K, T), zeros(K, T)],
    )
end


function update_motifs!(
    rule::ADMMUpdate, data, W, H; rhow=10, admm_W_maxiter=30, admm_tol=1e-4,
    nonnegW=true, kwargs...
)
    mu = 10
    tau = 2

    K, N, L = size(W)
    T = size(H, 2)
    loss_hist = []

    # Initialize auxilliary and dual variables
    Wstk_last = zeros(L*K, N)
    Wstk = zeros(L*K, N)
    Z1 = zeros(T, N)
    Z2 = zeros(L*K, N)
    Z3 = zeros(L*K, N)
    Z3_last = zeros(size(Z3))

    U1 = rule.U[1]
    U2 = rule.U[2]
    U3 = rule.U[3]
    U1 .= 0
    U2 .= 0
    U3 .= 0

    # Precompute reused quantities
    Hstk = shift_and_stack(H, L)
    HstkT = copy(Hstk')
    HHTI_fact = cholesky((Hstk * Hstk') + (2*I)(L*K))
    
    estT = zeros(T, N)
    dataT = copy(data')
    V2 = zeros(size(U2))
    rhs = zeros(K*L, N)
    weights = zeros(1, N)

    for iter = 1:admm_W_maxiter
        # Update W itself
        mul!(rhs, Hstk, Z1-U1)
        Wstk_last .= Wstk
        ldiv!(Wstk, HHTI_fact, rhs + Z2-U2 + Z3-U3)
    
        # Update wrt primary objective
        mul!(estT, HstkT, Wstk)
        @. Z1 = (1 / (1 + 1/rhow)) * ((estT + U1) + (1/rhow) * dataT)

        # Update wrt norm constraint
        @. V2 = Wstk + U2
        sum!(weights, V2 .^ 2)
        for n = 1:N
            if weights[n] >= 1
                @views @. Z2[:, n] = V2[:, n] / sqrt(weights[n])
            end
        end

        # Update wrt nonnegativity
        Z3_last .= Z3
        if nonnegW
            @. Z3 = max(0, Wstk+U3)
        else
            @. Z3 = Wstk + U3
        end

        # Update dual variable
        @. U1 += estT - Z1
        @. U2 += Wstk - Z2
        @. U3 += Wstk - Z3
        
        push!(
            loss_hist, 
            norm(dataT - HstkT * Z3) / rule.datanorm
        )

        # Check convergence
        if (length(loss_hist) > 1) 
            diff = loss_hist[end-1] - loss_hist[end]
            if diff < 0
                # Revert back
                @. Z3 = Z3_last
                loss_hist = loss_hist[1:end-1]
            end
            diff < admm_tol && break
        end
    end

    #plt.plot(loss_hist, ls="dashdot")
    #@show loss_hist[end]

    # Fold W
    for l = 1:L
        for n = 1:N
            for k = 1:K
                W[k, n, l] = Z3[(l-1)*K+k, n]
            end
        end
    end
end


function update_feature_maps!(
    rule::ADMMUpdate, data, W, H; 
    rhoh=10, admm_H_maxiter=30, l1H=0, admm_tol=1e-4, nonnegH=true, kwargs...
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

    loss_hist = []
    K, N, L = size(W)
    T = size(H, 2)

    #Initialize auxilliary and dual variables
    est = zeros(size(data))
    Z1 = zeros(size(data))  # Z1 = tensor_conv(W, H), handles loss function
    Z2 = zeros(size(H))  # Z2 = H, handles
    Z3_last = zeros(size(H))
    Z3 = zeros(size(H))  # Z3 = H

    Q1 = rule.Q[1]  # dual of Z1's equality constraint
    Q2 = rule.Q[2]  # dual of Z2's equality constraint
    Q3 = rule.Q[3]  # dual of Z3's equality constraint
    Q1 .= 0
    Q2 .= 0
    Q3 .= 0

    # Precompute and allocate reused quantities
    normW = norm(W)
    wh, whc, gram_arr = precompute_solveH(W, 1, T)

    bh = zeros(ComplexF64, size(Z1))
    v1h = zeros(ComplexF64, size(Z2))
    v2h = zeros(ComplexF64, size(Z3))
    hh = zeros(ComplexF64, K, T)
    esth = zeros(ComplexF64, N, T)
    beta = zeros(ComplexF64, K)

    true_est = zeros(size(data))

    for iter = 1:admm_H_maxiter
        # Update primal variable H
        fastsolveH!(
            H, gram_arr, whc, Z1-Q1, Z2-Q2, Z3-Q3, 1, bh, v1h, v2h, hh, beta
        )

        # Update Z1 (diagonal quadratic)
        tesnor_circconv!(est, whc, H, hh, esth)
        @. Z1 = (1 / (1 + 1/rhoh)) * ((est + Q1) + (1/rhoh) * data)

        # Update Z2 (shrinkage)
        @. Z2 = sign(H + Q2) * max(0, abs(H + Q2) - l1H/rhoh)

        # Update Z3 (projection)
        @. Z3_last = Z3
        if nonnegH
            @. Z3 = max(0, H + Q3)
        else
            @. Z3 = H + Q3
        end

        tensor_conv!(true_est, W, Z3)
        push!(loss_hist, norm(true_est - data) / rule.datanorm)

        # Check convergence
        if (length(loss_hist) > 1) 
            diff = loss_hist[end-1] - loss_hist[end]
            if diff < 0
                # Revert back
                @. Z3 = Z3_last
                loss_hist = loss_hist[1:end-1]
            end
            diff < admm_tol && break
        end

        # Update dual variable Q
        @. Q1 += est - Z1
        @. Q2 += H - Z2
        @. Q3 += H - Z3        
    end

    H .= Z3

    #plt.plot(loss_hist, ls="dotted")
    #@show loss_hist[end]
    #println("-")

    return norm(compute_resids(data, W, H)) / rule.datanorm
end


function precompute_solveH(W, rho, T)
    K, N, L = size(W)
    
    wh = zeros(ComplexF64, K, N, T)
    wh[:, :, 1:L] .= W

    fft!(wh, 3)
    whc = conj(wh)

    gram_arr = []
    for t = 1:T
        @views push!(
            gram_arr, 
            cholesky(whc[:, :, t] * whc[:, :, t]' + 2*rho*I(K))  # TODO drop whc
        )
    end

    return wh, whc, gram_arr
end

function fastsolveH!(
    H, gram_arr, whc, B, V1, V2, rho, bh, v1h, v2h, hh, beta
)
    K, N, T = size(whc)

    @. bh = B
    @. v1h = V1
    @. v2h = V2
    fft!(bh, 2)
    fft!(v1h, 2)
    fft!(v2h, 2)

    @inbounds for t = 1:T
        # Construct right hand side
        @views mul!(beta, whc[:, :, t], bh[:, t])
        @views @. beta += sqrt(rho) * (v1h[:, t] + v2h[:, t])

        # Solve
        @views ldiv!(hh[:, t], gram_arr[t], beta)
    end

    ifft!(hh, 2)
    @. H = real(hh)
end


#=
    R = conv(W, H)
    Λ = exp(R)
    y_{ij} \sim Poisson(Λ_{ij})

    J(Λ) = Σ_{ij} -y_{ij} R_{ij} + exp(R_{ij}) + ρ (R_{ij} - E_{ij})^2

    ∂_{ij}J = -y_{ij} + exp(R_{ij}) + 2 ρ (R_{ij} - E_{ij})
    0 = ...
    y_{ij} + 2 w E_{ij} = exp(R_{ij}) + 2 ρ R_{ij}
=#