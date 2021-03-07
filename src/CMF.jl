module CMF

export tensor_conv
export ConvolutionalFactorization
export AlternatingOptimizer, PGDUpdate

# Dependencies.
using LinearAlgebra
using HDF5
using FFTW

import Base: sortperm
import Random
import HDF5
import JLD
import MLJModelInterface
import ReverseDiff

# Constants
const Tensor{T} = Array{T, 3}

# Core files
include("atoms.jl")
include("./common.jl")
include("./model.jl")
#include("./visualize.jl")
#include("./evaluate.jl")

# Algorithms
include("./algs/alternating.jl")
include("./algs/mult.jl")
include("./algs/hals.jl")
# include("./algs/anls.jl")
include("./algs/admm.jl")
include("./algs/pgd.jl")
# include("./algs/separable.jl")



end

# TODO  make a test suite
# TODO  think about masks
# TODO  update the PGDUpdate struct
# TODO  transform
# TODO  move init rand and converged


# LATER...
# TODO  add sparse mask
# TODO  add general p-norm
# TODO  generalize nonnegativity to upper and lower bounds
# TODO  pyplot or plots
# TODO  consider travis ci
# ?     make eval a struct call
# TODO  allow setting initial W, H
# TODO  reincorporate separable stuff
# TODO  add wrapper that avoids MLJ business
# TODO  common constructors

# struct PGD end
#PGD() = AlternatingOptimizer(PGDUpdate())
#alg = AlternatingOptimizer(PGDUpdate())