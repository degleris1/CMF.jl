module CMF

export fit_cnmf

# Dependencies.
using LinearAlgebra
using HDF5

import PyPlot
import Base: sortperm
import Random
import JLD

const plt = PyPlot

# Constants
const EPSILON = eps()

# Need to load model.jl dependencies
# before including model.jl
include("./algs/mult.jl")
include("./algs/hals.jl")
include("./algs/anls.jl")
include("./algs/separable.jl")

# Source files.
include("./common.jl")
include("./model.jl")
include("./visualize.jl")
include("./evaluate.jl")

end
