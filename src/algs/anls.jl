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
    nnls_alg: julia symbol, one of the following:

                :pivot : Pivot method with caching defined in NonNegLeastSquares.jl. Updates
                         a single column of H at once.

                :pivot_block : Pivot method updating multiple columns of H at once.
                               Maybe give better performance for some problems.

                :LBFGS : Uses FORTRAN wrapper of the LBFGS-B solver for general box-constrained optimization.
                         Updates a single row of W and a single column of H at a time.
"""
function update!(data, W, H, meta; nnls_alg=:pivot, kwargs...)
    if (meta == nothing)
        meta = ANLSmeta(data, W, H)
    end

    if nnls_alg == :pivot
        pivot_update_H_cols!(W, H, meta)
        pivot_update_W!(data, W, H)

    elseif nnls_alg == :pivot_block
        pivot_block_update_H!(W, H, meta)
        pivot_update_W!(data, W, H)

    elseif nnls_alg == :LBFGS
        println("using LBFGS")
        LBFGSB_update_H!(W, H, meta)
        LBFGSB_update_W!(data, W, H)

    else
        error("NNLS alg ", nnls_alg, "is not supported")
    end

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
