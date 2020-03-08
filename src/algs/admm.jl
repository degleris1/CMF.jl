"""
Fit CMF using alternating optimization, where each update is solved
using ADMM.
"""

mutable struct ADMMUpdate <: AbstractCFUpdate
    datanorm
    loss_histories
    U
end


function ADMMUpdate(data::Matrix, W::Tensor, H::Matrix)
    return ADMMUpdate(norm(data), [], [])
end


function update!(
    rule::ADMMUpdate, data, W, H;
    rhow=4, rhoh=2, beta=0, 
    admm_W_maxiter=10, admm_H_maxiter=15,
    kwargs...
)
    L, N, K = size(W)
    T = size(H, 2)

    # W update
    # ---

    # Precompute reused quantities
    Hstk = shift_and_stack(H, L)
    HHTI = (Hstk * Hstk') + (2*I)(L*K)

    # Initialize auxilliary and dual variables
    loss_hist = []
    gap_hist = zeros(admm_W_maxiter, 3)
    Wstk = zeros(N, L*K)
    Z1 = zeros(N, T)
    Z2 = zeros(N, L*K)
    Z3 = zeros(N, L*K)

    if length(rule.U) == 0
        push!(rule.U, zeros(N, T))
        push!(rule.U, zeros(N, L*K))
        push!(rule.U, zeros(N, L*K))
    end

    U1 = rule.U[1]
    U2 = rule.U[2]
    U3 = rule.U[3]

    est = zeros(N, T)
    V2 = zeros(size(U2))
    #U1 = zeros(N, T)
    #U2 = zeros(N, L*K)
    #U3 = zeros(N, L*K)

    push!(loss_hist, norm(Wstk * Hstk - data) / rule.datanorm)
    for iter = 1:admm_W_maxiter
        # Update W itself
        Wstk .= (HHTI \ (Hstk * (Z1-U1)' + (Z2-U2)' + (Z3-U3)'))'

        # Update wrt primary objective
        est .= Wstk * Hstk
        Z1 .= (1 / (1 + 1/rhow)) * ((est + U1) + (1/rhow) * data)

        # Update wrt norm constraint
        V2 .= Wstk + U2
        for n = 1:N
            weight = norm(V2[n, :])
            if weight >= 1
                Z2[n, :] .= V2[n, :] / weight
            end
        end

        # Update wrt nonnegativity
        @. Z3 = max(0, Wstk+U3)

        # Update dual variable
        @. U1 += est - Z1
        @. U2 += Wstk - Z2
        @. U3 += Wstk - Z3

        #push!(loss_hist, norm(est - data) / rule.datanorm)
        #gap_hist[iter, :] = [norm(Wstk * Hstk - Z1) norm(Wstk - Z2) norm(Wstk - Z3)]
    end

    # println("ADMM ----")
    # println(norm(Wstk * Hstk - Z1))
    # println(norm(Wstk - Z2))
    # println(norm(Wstk - Z3))
    # println("---")

    # plt.plot(loss_hist, label=rhow, lw=1)
    # plt.plot(gap_hist[:, 1], label=string("Gap-", rhow), lw=1, linestyle="dotted")
    # plt.show()

    W[:, :, :] = fold_W(Z3, L, N, K)
    #push!(rule.loss_histories, loss_hist)




    # H update
    # ---
    num = tensor_transconv(W, data)
    denom = zeros(size(H))

    #loss_hist = [compute_loss(data, W, H)]
    for iter = 1:admm_H_maxiter
        denom .= tensor_transconv(W, tensor_conv(W, H))
        @. H *= num / (denom + beta + eps())
        #push!(loss_hist, compute_loss(data, W, H))
    end
    #plt.plot(loss_hist)
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

    return norm(compute_resids(data, W, H)) / rule.datanorm
end