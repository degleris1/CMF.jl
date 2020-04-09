mutable struct PGDUpdate <: AbstractCFUpdate
    datanorm
    gradW
    gradH
    est
    
    esth
    wh
    hh

    stepW
    stepH
    cur_loss
    step_incr
    step_decr
end


function PGDUpdate(data::Matrix, W::Tensor, H::Matrix)
    L, N, K = size(W)
    T = size(H, 2)

    datanorm = norm(data)
    est = tensor_conv(W, H)

    return PGDUpdate(
        datanorm,
        zeros(size(W)),
        zeros(size(H)),
        est,
        zeros(ComplexF64, size(data)),
        zeros(ComplexF64, K, N, T),
        zeros(ComplexF64, size(H)),

        5,
        5,
        datanorm,
        1.05,
        0.70
    )
end

function update_motifs!(rule::PGDUpdate, data, W, H; kwargs...)
    # Unpack dims
    K, T = size(H)
    K, N, L = size(W)

    # Define gradient function
    grad!(w, gw) = compute_gradW!(gw, w, H, data, rule.est)

    # Define projection function
    function proj!(w)
        @. w = max(0, w)
    end

    # Take projected gradient step
    pgd!(W, rule.gradW, grad!, proj!; step=rule.stepW)

    # Update step size and loss
    tensor_conv!(rule.est, W, H)
    loss = norm(rule.est - data)

    if loss < rule.cur_loss
        rule.stepW *= rule.step_incr
    else
        rule.stepW *= rule.step_decr
    end
    rule.cur_loss = loss
end

function update_feature_maps!(rule::PGDUpdate, data, W, H; kwargs...)
    
    # Define gradient function
    grad!(h, gh) = compute_gradH!(gh, W, h, data, rule.est)

    # Define projection function
    function proj!(h)
        @. h = max(0, h)
    end

    # Take projected gradient step
    pgd!(H, rule.gradH, grad!, proj!, step=rule.stepH)

    # Update step size and loss
    tensor_conv!(rule.est, W, H)
    loss = norm(rule.est - data)

    if loss < rule.cur_loss
        rule.stepH *= rule.step_incr
    else
        rule.stepH *= rule.step_decr
    end
    rule.cur_loss = loss
    
    return loss / rule.datanorm
end


function pgd!(x, gradx, compute_gradx!, proj!; step=1)
    # Step 1: find gradient ∇_H J(W, H)
    compute_gradx!(x, gradx)

    # Step 2: compute step size, α = (a/i) / ||∇ J|| 
    alpha = step / norm(gradx)

    # Step 3: update W: descend and project
    @. x -= alpha * gradx
    proj!(x)
end


"""
Compute the gradient with respect to W

The variables -- est, -- are workspace variables.
"""
function compute_gradW!(gradW, W, H, data, est)
    # Unpack dimensions
    K, N, L = size(W)
    T = size(H, 2)

    # Compute the gradient of W, which is
    # dW J(W) = C(H)^T db̂ D(b, b̂) +  dW R(W)
    
    # for the square loss, this is
    # dW J(W) = C(H)^T (2b̂ - 2b) + dW R(W)
    #         = C(H)^T (2C(H)w - 2b) + dW R(W)

    # note that it is most efficient to precompute
    #  A = 2 C(H)^T C(H)
    #  d = 2 C(H)^T b
    # and then compute Aw - d at each gradient step

    # Compute b̂
    #tensor_conv!(est, W, H)

    # Compute difference, r = HW - b
    @. est -= data

    # Compute the transpose convolution, 2 * C(H)^T r
    for lag = 0:(L-1)
        @views mul!(gradW[:, :, lag+1], shift_cols(H, lag), est[:, 1+lag:T]')
    end
    @. gradW *= 2

    # Add the gradient with respect to regularizers
    # -- No regularizers --
end


"""
Compute the gradient with respect to H

The variables -- est, esth, wh, hh -- are workspace variables.
"""
function compute_gradH!(gradH, W, H, data, est)
    # Compute the gradient of W, which is
    # dW J(H) = W^T db̂ D(b, b̂) +  dW R(H)
    
    # for the square loss, this is
    # dW J(H) = C(W)^T (2b̂ - 2b) + dW R(H)
    #         = C(W)^T (2Wh - 2b) + dW R(H)

    # note that it is most efficient to precompute
    #  A = 2 C(W)^T W
    #  d = 2 C(W)^T b
    # and then compute Ah - d at each gradient step

    # Compute b̂
    #tensor_conv!(est, W, H)
    #cconv!(est, nothing, H, esth, wh, hh; compute_wh=false)

    # Compute difference, r = Wh - b
    @. est -= data

    # Compute the transpose convolution, ∇ = 2 * C(W)^T r
    tensor_transconv!(gradH, W, est)
    @. gradH *= 2


    # Add the gradient with respect to regularizers
    # -- No regularizers --
end





function cconv!(est, Wpad, H, esth, wh, hh; compute_hh=true, compute_wh=true)
    K, N, T = size(wh)

    if compute_hh
        @. hh = H
        fft!(hh, 2)
    end

    if compute_wh
        @. wh = Wpad
        fft!(Wpad, 3)
    end

    _cconv!(esth, wh, hh)

    ifft!(esth, 2)
    @. est = real(esth)
end

function _cconv!(esth, wh, hh)
    K, N, T = size(wh)
    for t = 1:T
        for n = 1:N
            @views esth[n, t] = transpose(wh[:, n, t]) * hh[:, t] 
        end
    end
end