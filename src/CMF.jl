module CMF

export fit_cnmf, init_rand, evaluate_feature_maps
export MultUpdate, HALSUpdate, ANLSUpdate, ADMMUpdate

# Dependencies.
using LinearAlgebra
using HDF5
using NonNegLeastSquares

import PyPlot
import Base: sortperm
import Random
import HDF5
import JLD

const plt = PyPlot

# Constants
const EPSILON = eps()
const Tensor{T} = Array{T, 3}

# Core files
include("./common.jl")
include("./model.jl")
include("./visualize.jl")
include("./evaluate.jl")

# Algorithms
include("./algs/alternating.jl")
include("./algs/mult.jl")
include("./algs/hals.jl")
include("./algs/anls.jl")
include("./algs/admm.jl")
# include("./algs/separable.jl")



end
