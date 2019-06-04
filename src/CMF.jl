module CMF

export fit_cnmf

# Dependencies.
import PyPlot; const plt = PyPlot
using LinearAlgebra
using HDF5
import Base: sortperm
import Random

const ds = Distributions

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
