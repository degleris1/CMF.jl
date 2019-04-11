module CMF

# Dependendies.
using LinearAlgebra
using HDF5
using ImageFiltering
import DSP
import WAV

# Source files.
include("./common.jl")

include("./mult.jl")
include("./hals.jl")
# include("./anls.jl") 

include("./model.jl")

include("./datasets.jl")
include("./visualize.jl")

end