module ANLS
"""
Fit CNMF using Alternating Non-Negative Least Squares.
"""

# Import 
using LinearAlgebra
include("../common.jl")
include("./nnls/pivot.jl")
include("./nnls/lbfgsb.jl")

"""
Main update rule for ANLS

Params
------
    data: array, data matrix
    W, H: array, components of model at current iteration.
    meta: struct storing residuals and norm of data matrix
    H_solver: julia symbol, one of the following:

                :pivot : Pivot method with caching defined in NonNegLeastSquares.jl. Updates
                         a single column of H at once.

                :pivot_block : Pivot method updating multiple columns of H at once.
                               Maybe give better performance for some problems.

                :LBFGS : Uses FORTRAN wrapper of the LBFGS-B solver for general box-constrained optimization.
                         Updates a single row of W and a single column of H at a time.
"""

H_algs = Dict(
    :pivot_block => pivot_block_update_H!,
    :pivot => pivot_update_H_cols!,
    :LBFGSB => LBFGSB_update_H!
)

W_algs = Dict(
    :pivot => pivot_update_W!,
    :LBFGSB => LBFGSB_update_W!
)
function update!(data, W, H, meta; 
                 H_solver=:pivot_block, 
                 W_solver=:LBFGSB,
                 kwargs...
                )

    if (meta == nothing)
        meta = ANLSmeta(data, W, H)
    end

    if !(H_solver in keys(H_algs))
        error("Invalid argument for H_solver:", H_solver)
    end
    if !(W_solver in keys(W_algs))
        error("Invalid argument for W_solver:", H_solver)
    end

    # Call H and W updates using specified solvers
    H_algs[H_solver](W, H, meta)
    W_algs[W_solver](data, W, H)

    meta.resids = compute_resids(data, W, H)
    return norm(meta.resids) / meta.data_norm, meta
end


"""
Private
"""

mutable struct ANLSmeta
    resids
    data_norm
    function ANLSmeta(data, W, H)
        resids = compute_resids(data, W, H)
        data_norm = norm(data)
        return new(resids, data_norm)
    end
end

end  # module
