module CMF

# Dependendies.
import PyPlot; const plt = PyPlot
using LinearAlgebra
using HDF5
using ImageFiltering
import DSP
import WAV
import Distributions
import Base: sortperm

const ds = Distributions

# Need to load model.jl dependencies
# before including model.jl
include("./algs/mult.jl")
include("./algs/hals.jl")
include("./algs/anls.jl")

# Source files.
include("./common.jl")
include("./model.jl")

include("./datasets.jl")
include("./visualize.jl")

end
