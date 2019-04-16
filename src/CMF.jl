module CMF

# Dependendies.
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

include("./mult.jl")
include("./hals.jl")
# include("./anls.jl") 

include("./model.jl")

include("./datasets.jl")
include("./visualize.jl")

end