module CMF

# Dependendies.
import PyPlot
using LinearAlgebra
using HDF5
using ImageFiltering
import DSP
import WAV
import Distributions
import Base: sortperm

const ds = Distributions

# Source files.
include("./common.jl")
include("./model.jl")

include("./datasets.jl")
include("./visualize.jl")

end
