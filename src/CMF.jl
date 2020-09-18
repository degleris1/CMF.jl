module CMF

export fit_cnmf, init_rand
export MultUpdate, HALSUpdate, ADMMUpdate, PGDUpdate

# Dependencies.
using LinearAlgebra
using HDF5
using FFTW

import PyPlot
import Base: sortperm
import Random
import HDF5
import JLD

# Constants
const Tensor{T} = Array{T, 3}

# Core files
include("atoms.jl")
include("./common.jl")
include("./model.jl")
include("./visualize.jl")
include("./evaluate.jl")

# Algorithms
include("./algs/alternating.jl")
include("./algs/mult.jl")
include("./algs/hals.jl")
# include("./algs/anls.jl")
include("./algs/admm.jl")
include("./algs/pgd.jl")
# include("./algs/separable.jl")



end
