"""
atoms.jl contains the AbstractLossFunction, AbstractPenalty, and AbstractConstraint
interfaces, as well as any loss functions, penalties, and constraints that
implement them.
"""


notyetimplemented() = error("Not yet implemented.")

abstract type AbstractLossFunction end
abstract type AbstractPenalty end
abstract type AbstractConstraint end


"""General loss function"""
AbstractLossFunction

"""Computes the gradient of `D`(`est`, `data`) and stores it in `grad`."""
grad!(D::AbstractLossFunction, grad, est, data) = notyetimplemented()

"""
Computes prox_{`D`(`est`, `data`), `alpha`}(`v`) and stores it in `out`.
Here, `est` is the variable of the prox operator.
"""
prox!(D::AbstractLossFunction, out, est, data, v, alpha) = notyetimplemented()


"""General penalizer function."""
AbstractPenalty

weight(p::AbstractPenalty) = p.weight

""" Computes the gradient of `P`(`x`) and **adds** it to `g`. """
grad!(P::AbstractPenalty, g, x) = notyetimplemented()
prox!(P::AbstractPenalty, x) = notyetimplemented()


"""General constraints."""
AbstractPenalty

projection!(c::AbstractConstraint, x) = notyetimplemented()
projection!(c::Nothing, x) = 0
prox!(c::AbstractConstraint, x) = projection!(c, x)


########
####    LOSS FUNCTIONS
########


"""D(b, b̂) = || b - b̂ ||_2^2"""
struct SquareLoss <: AbstractLossFunction end
function grad!(D::SquareLoss, grad, est, data)
    @. grad = 2 * (est - data)
end
function eval(D::SquareLoss, b, est)
    return norm(b - est)^2
end


"""D(b, b̂) = || b - b̂ ||_1"""
struct AbsoluteLoss <: AbstractLossFunction end
function grad!(D::AbsoluteLoss, grad, est, data)
    @. grad = sign(est - data)
end
function eval(D::AbsoluteLoss, b, est)
    return norm(b - est, 1)
end


"""Docstring"""
struct MaskedLoss <: AbstractLossFunction
    loss::AbstractLossFunction
    mask
end
function grad!(D::MaskedLoss, grad, est, data)
    grad!(D.loss, grad, est, data)
    @. grad *= D.mask
end
function eval(D::MaskedLoss, b, est)
    return eval(D.loss, D.mask .* b, D.mask .* est)
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


""" R(x) = ||x||_1 """
struct AbsolutePenalty <: AbstractPenalty
    weight
end
function grad!(P::AbsolutePenalty, g, x)
    @. g += P.weight * sign(x)
end


########
####    CONSTRAINTS
########


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