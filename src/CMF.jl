module CMF

export fit_cnmf

# Dependencies.
using LinearAlgebra
using HDF5

import PyPlot
import Base: sortperm
import Random
import HDF5
import JLD

const plt = PyPlot

# Constants
const EPSILON = eps()
const Tensor{T} = Array{T, 3}

# Need to load model.jl dependencies
# before including model.jl
include("./common.jl")

include("./algs/alternating.jl")
include("./algs/mult.jl")
# include("./algs/hals.jl")
# include("./algs/anls.jl")
# include("./algs/separable.jl")

# # Source files.

include("./model.jl")
include("./visualize.jl")
include("./evaluate.jl")

end
