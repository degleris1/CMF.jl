
abstract type AbstractLossFunction end

""" Computes the gradient of `D`(`est`, `data`) and stores it in `grad`. """
grad!(D::AbstractLossFunction, grad, est, data) = error("Must implement gradient.")

"""
Computes prox_{`D`(`est`, `data`), `alpha`}(`v`) and stores it in `out`.
Here, `est` is the variable of the prox operator.
"""
prox!(D::AbstractLossFunction, out, est, data, v, alpha) = error(
    "Must implement proximal operator."
)


""" All penalties must have the field `weight`. """
abstract type AbstractPenalty end

""" Computes the gradient of `P`(`x`) and **adds** it to `g`. """
grad!(P::AbstractPenalty, g, x) = error("Must implement gradient.")
prox!(P::AbstractPenalty, x) = error("Must implement proximal operator.")


abstract type AbstractConstraint end
projection!(c::AbstractConstraint, x) = error("Must implement the projection operator.")
projection!(c::Nothing, x) = 0
prox!(c::AbstractConstraint, x) = projection!(c, x)


""" D(b, hat b) = || b - hat b ||_2^2 """
struct SquareLoss <: AbstractLossFunction end
function grad!(D::SquareLoss, grad, est, data)
    @. grad = 2 * (est - data)
end


""" D(b, hat b) = || b - hat b ||_2^2 """
struct AbsoluteLoss <: AbstractLossFunction end
function grad!(D::AbsoluteLoss, grad, est, data)
    @. grad = sign(est - data)
end


""" R(x) = ||x||_2^2 """
struct SquarePenalty <: AbstractPenalty
    weight
end
function grad!(P::SquarePenalty, g, x)
    @. g += 2 * P.weight * x
end


""" R(x) = ||x||_1 """
struct AbsolutePenalty <: AbstractPenalty
    weight
end
function grad!(P::AbsolutePenalty, g, x)
    @. g += P.weight * sign(x)
end


""" x_i >= 0 for all i """
struct NonnegConstraint <: AbstractConstraint end
function projection!(c::NonnegConstraint, x)
    @. x = max(eps(), x)
end


mutable struct PGDUpdate <: AbstractCFUpdate
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
    dims = (K, N, L, T)

    datanorm = norm(data)
    est = tensor_conv(W, H)

    return PGDUpdate(
        dims,
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


function update_motifs!(
    rule::PGDUpdate, data, W, H;
    loss_func=SquareLoss(),
    constrW=NonnegConstraint(),
    penaltiesW=[SquarePenalty(1)],
    kwargs...
)
    # Define gradient function
    grad!(gradw, w, est) = compute_gradW!(gradw, H, est, rule.dims)
    
    # Define projection function
    proj!(w) = projection!(constrW, w)

    # Take projected gradient step
    newstep = pgd!(
        W, rule.gradW, grad!, proj!, rule.stepW,
        rule, data, W, H, loss_func, penaltiesW,
    )
    rule.stepW = newstep
end


function update_feature_maps!(
    rule::PGDUpdate, data, W, H; 
    loss_func=SquareLoss(),
    constrH=NonnegConstraint(),
    penaltiesH=[],
    kwargs...
)
    
    # Define gradient function
    grad!(gradh, h, est) = compute_gradH!(gradh, W, est)

    # Define projection function
    proj!(h) = projection!(constrH, h)

    # Take projected gradient step
    newstep = pgd!(
        H, rule.gradH, grad!, proj!, rule.stepH,
        rule, data, W, H, loss_func, penaltiesH,
    )
    rule.stepH = newstep

    return rule.cur_loss / rule.datanorm
end


""" Compute the gradient with respect to W -- dW J(W) = C(H)^T db̂. """
function compute_gradW!(gradW, H, est, dims)
    # Unpack dimensions
    K, N, L, T = dims

    # Compute the transpose convolution, C(H)^T r
    for lag = 0:(L-1)
        @views mul!(gradW[:, :, lag+1], shift_cols(H, lag), est[:, 1+lag:T]')
    end
end


""" Compute the gradient with respect to H. """
function compute_gradH!(gradH, W, est)
    # Compute the transpose convolution, ∇ = C(W)^T r
    tensor_transconv!(gradH, W, est)
end


function pgd!(
    x, gradx, compute_gradx!, proj!, step,
    r::PGDUpdate, data, W, H, loss_func, penalties,
)
    # Step 1: find gradient ∇_H J(W, H)
    grad!(loss_func, r.est, r.est, data)  # Compute db̂ D(b b̂)
    compute_gradx!(gradx, x, r.est)  # Compute dx db̂
    for pen in penalties
        grad!(pen, gradx, x)
    end

    # Step 2: compute step size, α = (a/i) / ||∇ J|| 
    alpha = step / (norm(gradx) + eps())

    # Step 3: update W: descend and project
    @. x -= alpha * gradx
    proj!(x)

    # Step 4: eval
    # Update step size and loss
    tensor_conv!(r.est, W, H)
    loss = norm(r.est - data)

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