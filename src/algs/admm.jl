"""
Fit CMF using alternating optimization, where each update is solved
using ADMM.
"""

mutable struct ADMMUpdate <: AbstractCFUpdate
    datanorm
    U
end


function ADMMUpdate(data::Matrix, W::Tensor, H::Matrix)
    L, N, K = size(W)
    T = size(H, 2)
    return ADMMUpdate(
        norm(data),
        [zeros(T, N), zeros(L*K, N), zeros(L*K, N)], 
    )
end


function update_motifs!(rule::ADMMUpdate, data, W, H; rhow=4, admm_W_maxiter=15,
                        fast=false, kwargs...)
    # L, N, K = size(W)
    # T = size(H, 2)

    # # Precompute reused quantities
    # Hstk = shift_and_stack(H, L)


    # HstkT = copy(Hstk')
    # HHTI = (Hstk * Hstk') + (2*I)(L*K)
    # if fast  # TODO why is this slower? Maybe compute cholesky?
    #     HHTI_inv = inv(HHTI)
    # end

    # # Initialize auxilliary and dual variables
    # loss_hist = []
    # gap_hist = zeros(admm_W_maxiter, 3)
    # Wstk = zeros(L*K, N)
    # Z1 = zeros(T, N)
    # Z2 = zeros(L*K, N)
    # Z3 = zeros(L*K, N)

    # U1 = rule.U[1]
    # U2 = rule.U[2]
    # U3 = rule.U[3]

    # estT = zeros(T, N)
    # dataT = copy(data')
    # V2 = zeros(size(U2))

    # for iter = 1:admm_W_maxiter
    #     # Update W itself
    #     if fast
    #         Wstk .= HHTI_inv * (Hstk * (Z1-U1) + (Z2-U2) + (Z3-U3))
    #     else
    #         Wstk .= HHTI \ (Hstk * (Z1-U1) + (Z2-U2) + (Z3-U3))
    #     end

    #     # Update wrt primary objective
    #     estT .= HstkT * Wstk
    #     Z1 .= (1 / (1 + 1/rhow)) * ((estT + U1) + (1/rhow) * dataT)

    #     # Update wrt norm constraint
    #     @. V2 = Wstk + U2
    #     for n = 1:N
    #         weight = norm(V2[:, n])
    #         if weight >= 1
    #             @. Z2[:, n] = V2[:, n] / weight
    #         end
    #     end

    #     # Update wrt nonnegativity
    #     @. Z3 = max(0, Wstk+U3)

    #     # Update dual variable
    #     @. U1 += estT - Z1
    #     @. U2 += Wstk - Z2
    #     @. U3 += Wstk - Z3
    # end

    # W .= fold_W(Z3', L, N, K)

    # First do mult updates
    L, N, K = size(W)
    T = size(H, 2)
    num = zeros(L, N, K)
    denom = zeros(L, N, K)
    est = zeros(N, T)
    
    # Precompute numerator
    for lag = 0:(L-1)
        num[lag+1, :, :] .= data[:, 1+lag:T] * shift_cols(H, lag)'
    end

    # Repeatedly update W
    for iter in 1:admm_W_maxiter
        est .= tensor_conv(W, H)

        for lag = 0:(L-1)
            denom[lag+1, :, :] .= est[:, 1+lag:T] * shift_cols(H, lag)'
        end

        @. W *= num / (denom + eps())
    end

    # Renormalize factors
    for k = 1:K
        weight = norm(W[:, :, k])
        if weight >= 1
            @. W[:, :, k] /= weight
            @. H[k, :] *= weight
        end
    end
end


function update_feature_maps!(rule::ADMMUpdate, data, W, H; rhoh=2, admm_H_maxiter=10, 
                              l1H=0, kwargs...)
    # --
    num = tensor_transconv(W, data)
    denom = zeros(size(H))

    for iter = 1:admm_H_maxiter
        denom .= tensor_transconv(W, tensor_conv(W, H))
        @. H *= num / (denom + l1H + eps())
    end

    return norm(compute_resids(data, W, H)) / rule.datanorm
end


# function update!(
#     rule::ADMMUpdate, data, W, H;
#     rhow=4, rhoh=2, beta=0, 
#     admm_W_maxiter=10, admm_H_maxiter=15,
#     kwargs...
# )
    # Cache important stuff

    # Initialize auxilliary and dual variables
    # nH = rand(size(H)...)
    # Z1 = zeros(size(data))
    # Z2 = zeros(size(H))
    # Z3 = zeros(size(H))  

    # Q1 = zeros(size(Z1))
    # Q2 = zeros(size(Z2))
    # Q3 = zeros(size(Z3))

    # grad = zeros(size(nH))
    # loss_hist = [norm(tensor_conv(W, nH) - data) / rule.datanorm]

    # normW = norm(W)
    # for iter = 1:admm_maxiter
    #     # Update Z1 
    #     est = tensor_conv(W, nH)
    #     Z1 .= (1 / (1 + 1/rhoh)) * ((est + U1) + (1/rhoh) * data)

    #     # Update Z2 
    #     Z2 .= sign.(nH+Q2) .* max.(0, abs.(nH+Q2) .- beta/rhoh)

    #     # Update Z3
    #     Z3 .= max.(0, nH + Q3)

    #     # Update dual variable Q 
    #     Q1 .+= est - Z1
    #     Q2 .+= nH - Z2
    #     Q3 .+= nH - Z3

    #     linterm = 2*(tensor_transconv(W, Z1-Q1) + (Z2-Q2) + (Z3-Q3))
    #     for giter = 1:grad_maxiter
    #         grad .= (
    #             2 * (tensor_transconv(W, tensor_conv(W, nH)) + 2*nH)
    #             - linterm
    #         )
    #         nH .-= (0.5 * 0.95^giter / max(1, norm(grad))) * grad
    #     end
        
    #     push!(loss_hist, norm(tensor_conv(W, nH) - data) / rule.datanorm)
    # end

    # H[:, :] = nH

    #return norm(compute_resids(data, W, H)) / rule.datanorm
#end