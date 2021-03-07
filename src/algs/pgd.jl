

Base.@kwdef mutable struct PGDUpdate <: AbstractCFUpdate
    dims
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
    K, N, L = size(W)
    T = size(H, 2)
    datanorm = norm(data)
    return PGDUpdate(
        dims = (K, N, L, T),
        datanorm = datanorm,
        gradW = zeros(size(W)),
        gradH = zeros(size(H)),
        est = tensor_conv(W, H),
        
        # Fourier stuff. TODO: Fix these names.
        esth = zeros(ComplexF64, size(data)),
        wh = zeros(ComplexF64, K, N, T),
        hh = zeros(ComplexF64, size(H)),

        stepW = 5,
        stepH = 5,
        cur_loss = datanorm,
        step_incr = 1.05,
        step_decr = 0.70
    )
end


function update_motifs!(
    rule::PGDUpdate,
    model::ConvolutionalFactorization,
    data,
    W,
    H;
    kwargs...
)

    # Define gradient function
    loss(W) = evaluate_loss(model, W, H, data)
    ReverseDiff.gradient!(rule.gradW, loss, W)

    # Define projection function
    proj!(w) = projection!(model.W_constraint, w)

    # Create functions used by inner pgd update
    update!(est, W) = tensor_conv!(est, W, H)
    eval_loss(est) = evaluate_loss(model, W, H, data, est)

    # Instead of passing a function to compute the gradient,
    # here we're passing the gradient itself.
     newstep = _pgd!(
        W, rule.gradW,
        proj!, update!, eval_loss,
        rule.stepW, rule
    )

    # x,
    # gradx,
    # proj!::Function,
    # update!::Function,
    # eval_loss::Function,
    # step,
    # r::PGDUpdate,

    rule.stepW = newstep
end


function update_feature_maps!(
    rule::PGDUpdate,
    model,
    data,
    W,
    H;
    kwargs...
)
    
    # Define gradient function
    loss(H) = evaluate_loss(model, W, H, data)
    ReverseDiff.gradient!(rule.gradH, loss, H)

    # Define projection function
    proj!(h) = projection!(model.H_constraint, h)

    # Create functions used by inner pgd update
    update!(est, H) = tensor_conv!(est, W, H)
    eval_loss(est) = evaluate_loss(model, W, H, data, est)

    # Take projected gradient step
    newstep = _pgd!(
        H, rule.gradH,
        proj!, update!, eval_loss,
        rule.stepH, rule
    )
    rule.stepH = newstep

    return rule.cur_loss
end


function _pgd!(
    x,
    gradx,
    proj!::Function,
    update!::Function,
    eval_loss::Function,
    step,
    r::PGDUpdate,
)

    # Step 2: compute step size, α = (a/i) / ||∇ J|| 
    alpha = step / (norm(gradx) + eps())

    # Step 3: update W: descend and project
    @. x -= alpha * gradx
    proj!(x)

    # Step 4: eval
    # Update step size and loss
    update!(r.est, x)
    loss = eval_loss(r.est)

    if loss < r.cur_loss
        step *= r.step_incr
    else
        step *= r.step_decr
    end
    r.cur_loss = loss
    
    return step
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


##### DEPRECATED #####

# """ Compute the gradient with respect to W -- dW J(W) = C(H)^T db̂. """
# # function compute_gradW!(gradW, H, est, dims)
# #     # Unpack dimensions
# #     K, N, L, T = dims

# #     # Compute the transpose convolution, C(H)^T r
# #     for lag = 0:(L-1)
# #         @views mul!(gradW[:, :, lag+1], shift_cols(H, lag), est[:, 1+lag:T]')
# #     end
# # end

# function compute_gradW!(gradW, model, H, est)

# end


# """ Compute the gradient with respect to H. """
# function compute_gradH!(gradH, W, est)
#     # Compute the transpose convolution, ∇ = C(W)^T r
#     tensor_transconv!(gradH, W, est)
# end

