"""
atoms.jl contains the AbstractLoss, AbstractPenalty, and AbstractConstraint
interfaces, as well as any loss functions, penalties, and constraints that
implement them.
"""


notyetimplemented() = error("Not yet implemented.")

abstract type AbstractLoss end
abstract type AbstractPenalty end
abstract type AbstractConstraint end


"""General loss function"""
AbstractLoss

"""Computes the gradient of `D`(`est`, `data`) and stores it in `grad`."""
grad!(D::AbstractLoss, grad, est, data) = notyetimplemented()

"""
Computes prox_{`D`(`est`, `data`), `alpha`}(`v`) and stores it in `out`.
Here, `est` is the variable of the prox operator.
"""
prox!(D::AbstractLoss, out, est, data, v, alpha) = notyetimplemented()


"""General penalizer function."""
AbstractPenalty

weight(p::AbstractPenalty) = p.weight

""" Computes the gradient of `P`(`x`) and **adds** it to `g`. """
grad!(P::AbstractPenalty, g, x) = notyetimplemented()
prox!(P::AbstractPenalty, x) = notyetimplemented()
evaluate_loss(P::AbstractPenalty, x) = notyetimplemented()


"""General constraints."""
AbstractConstraint

projection!(c::AbstractConstraint, x) = notyetimplemented()
projection!(c::Nothing, x) = 0
prox!(c::AbstractConstraint, x) = projection!(c, x)

"""General function for evaluating a loss function.
   Takes a single loss function and a list of AbstractPenalty, 
   calls eval on each, and adds them. """
function evaluate_loss(
    D::AbstractLoss,
    W_penalties::Vector{AbstractPenalty},
    H_penalties::Vector{AbstractPenalty},
    W,
    H,
    b,
    est
)
    loss = evaluate_loss(D, b, est)
    loss += sum(Float64[evaluate_loss(p, W) for p in W_penalties])
    loss += sum(Float64[evaluate_loss(p, H) for p in H_penalties])
    return loss
end

function evaluate_loss(
    D::AbstractLoss,
    W_penalties::Vector{AbstractPenalty},
    H_penalties::Vector{AbstractPenalty},
    W,
    H,
    b
)
    est = tensor_conv(W, H)
    return evaluate_loss(D, W_penalties, H_penalties, W, H, b, est)
end


########
####    LOSS FUNCTIONS
########


"""D(b, b̂) = || b - b̂ ||_2^2"""
struct SquareLoss <: AbstractLoss end
function grad!(D::SquareLoss, grad, est, data)
    @. grad = 2 * (est - data)
end
function evaluate_loss(D::SquareLoss, b, est)
    return norm(b - est)^2
end


"""D(b, b̂) = || b - b̂ ||_1"""
struct AbsoluteLoss <: AbstractLoss end
function grad!(D::AbsoluteLoss, grad, est, data)
    @. grad = sign(est - data)
end
function evaluate_loss(D::AbsoluteLoss, b, est)
    return norm(b - est, 1)
end


"""Masked loss """
struct MaskedLoss <: AbstractLoss
    loss::AbstractLoss
    mask
end
function grad!(D::MaskedLoss, grad, est, data)
    grad!(D.loss, grad, est, data)
    @. grad *= D.mask
end
function evaluate_loss(D::MaskedLoss, b, est)
    return evaluate_loss(D.loss, D.mask .* b, D.mask .* est)
end


########
####    PENALTIES
########


""" R(x) = ||x||_2^2 """
struct SquarePenalty <: AbstractPenalty
    weight
end
function grad!(P::SquarePenalty, g, x)
    @. g += 2 * P.weight * x
end
function evaluate_loss(P::SquarePenalty, x)
    return P.weight * norm(x)^2
end
    


""" R(x) = ||x||_1 """
struct AbsolutePenalty <: AbstractPenalty
    weight
end
function grad!(P::AbsolutePenalty, g, x)
    @. g += P.weight * sign(x)
end
function evaluate_loss(P::AbsolutePenalty, x)
    return P.weight * norm(x, 1)
end


########
####    CONSTRAINTS
########
struct NoConstraint <: AbstractConstraint end
function projection!(c::NoConstraint, x)
    return x
end


""" x_i >= 0 for all i """
struct NonnegConstraint <: AbstractConstraint end
function projection!(c::NonnegConstraint, x)
    @. x = max(eps(), x)
end


""" || x ||_2 <= 1 """
struct UnitNormConstraint <: AbstractConstraint end
function projection!(c::UnitNormConstraint, x)
    M = size(x, 1)
    for m = 1:M
        xm = selectdim(x, 1, m)
        mag = norm(xm)
        if mag > 1
            @. xm /= mag
        end
    end
end
