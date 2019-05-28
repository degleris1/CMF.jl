module Separable

using NonNegLeastSquares
using Combinatorics
using LinearAlgebra
import PyPlot; plt = PyPlot

include("../common.jl")
include("./hals.jl")
include("./anls.jl")


""" Fit using the LCS Algorithm. """
function fit(data, K, L; thresh=0, verbose=false, refit_H=false,
             refit_W=false, refit_H_itr=10, spectral=true, kwargs...)
    N, T = size(data)

    # Step 1: successive projection to locate the columns of W
    V, vertices = SPA(data, K*L, thresh=thresh)

    # Step 2: compute unconstrained H (NMF)
    G = nonneg_lsq(V, data, alg=:pivot, variant=:comb)
    
    # Step 3: group rows of H to produce convolutive H
    groups = shift_cluster(G, K, L, verbose, spectral)

    # Step 4: sort rows of H
    for k = 1:K
        groups[k] = sort_group(groups[k], G)
    end
        
    # Create W, H based on grouping
    W = zeros(L, N, K)
    for k = 1:K
        W[:, :, k] = V[:, groups[k]]'
    end

    reps = [groups[k][1] for k in 1:K]
    H = G[reps, :]

    # Refit H with HALS
    meta = HALS.HALSMeta(data, W, H)
    HALS._setup_H_update!(W, H, meta)
    if (refit_H)
        for itr in 1:refit_H_itr
            HALS._update_H_regular!(W, H, meta.resids, meta.Wk_list, meta.W_norms, 0, 0)
        end
    end

    # Refit W with ANLS
    if (refit_W)
        ANLS._update_W!(data, W, H)
    end    
     
    return W, H
end


"""
SORT STEP
"""


""" Sort rows of G witin some group. """
function sort_group(group, G)
    G_gr = G[group, :]
    L = length(group)

    M = zeros(L, L)
    for i = 1:L
        for j = 1:L
            M[i, j] = arg_shift_max(G_gr[i, :], G_gr[j, :], L)
        end
    end

    weight = sum(M, dims=2)    
    return group[sort(collect(1:L), by=j->-weight[j])]
end


function arg_shift_max(h1, h2, L)
    argmax = 0
    max = 0

    for l in 0:L-1
        left = cosL(h1, h2, l, "a")
        right = cosL(h1, h2, l, "b")

        if (left > max)
            max = left
            argmax = l
        end
        if (right > max)
            max = right
            argmax = -l
        end
    end

    return argmax
end


"""
CLUSTER STEP
"""


""" Cluster based on shift distance. """
function shift_cluster(Ho, K, L, verbose, spectral)
    R, T = size(Ho)

    # Step 1: compute distance matrix
    dmat = zeros(R, R)
    for r = 1:R
        for p = r:R
            dmat[r, p] = shift_cos(Ho[r, :], Ho[p, :], L)
            dmat[p, r] = dmat[r, p]
        end
    end
    
    # Step 2: compute groups
    if (spectral)
        groups = find_groups_spectral(dmat, K, L)
    else
        groups = find_groups(dmat, K, L)
    end
    
    # Plot heatmap of similarity matrix and grouping
    if (verbose)
        fig, ax = plt.subplots()
        im = ax.imshow(dmat)
        ax.set_yticks(0:R-1)
        ax.set_yticklabels(invert_grouping(groups, R, K))

        cbar = ax.figure.colorbar(im, ax=ax)
        plt.title("Angles")

        plt.show()
    end

    return groups
end


function invert_grouping(groups, R, K)
    labels = []

    for r in 1:R
        for k in 1:K
            if (r in groups[k])
                push!(labels, k)
            end
        end
    end
    
    return labels
end


function find_groups(dmat, K, L)
    groups = [[] for k in 1:K]
    ungrouped = collect(1:L*K)
    
    for k in 1:K
        # Push a remaining element
        push!(groups[k], pop!(ungrouped))

        while (length(groups[k]) < L)
            # Add the element closest to the group
            sims = sum(dmat[groups[k], ungrouped], dims=1)
            _, i = findmax(sims)
            i = i[2]

            push!(groups[k], ungrouped[i])
            deleteat!(ungrouped, i)
        end
    end

    return groups
end


function find_groups_spectral(
    simat::Array{Float64, 2},
    K::Int,
    L::Int;
    binarize::Bool=false,
    verbose::Bool=true
)
    R = K * L

    # Demean
    simat = max.(0, simat .- (sum(simat) / R^2))

    if (binarize)  # Largest K*L^2 entries to 1, all else to zero
        largest = sort(1:R^2, by=j->simat[j], rev=true)
        simat = zeros(R, R)
        simat[largest[1:K*L^2]] .= 1
    end
    
    F = eigen(simat)
    lambda = F.values
    V = F.vectors

    rows = collect(1:R)
    free = (rows .!= 0)
    groups = []
    
    for k in 0:K-1
        p = R-k
        if (abs(maximum(V[:, p])) < abs(minimum(V[:, p])))
            V[:, p] = -V[:, p]  # Reorient
        end

        priority = sort(rows[free], by=j->V[j, p], rev=true)
        push!(groups, priority[1:L])
        free[priority[1:L]] .= false
    end

    if (verbose)
        plt.figure()
        plt.imshow(simat)
        plt.colorbar()
        plt.show()


        plt.figure()
        plt.plot(1:R, lambda, marker=".")
        plt.axvline(R - K + 1)
        plt.show()

        plt.figure()
        plt.plot(V[:, R-K+1:R], marker=".")
        plt.legend()
        plt.show()
    end

    return groups
end



"""
LOCATE STEP
"""


""" Successive projection algorithm. """
function SPA(data, K; thresh=0)
    col1 = colnorms(data, 1)
    col2 = colnorms(data, 2)
    DX = diagscale(col1)
    X = data * inv(DX)

    # Eliminate
    for j in 1:size(data, 2)
        if (col1[j] < thresh)
            X[:, j] .= 0
        end
    end
    
    vertices = []
    resid = X
    
    for r = 1:K
        _, jset = findsetmax(colnorms(resid))

        if (length(jset) == 1)
            j = jset[1]
        else  # Break ties
            _, k = findmax(col2[jset])
            j = jset[k]
        end
        push!(vertices, j)

        w = resid[:, j]
        resid = (I - (w * w' / norm(w)^2)) * resid
    end
    
    return data[:, sort(vertices)], sort(vertices)
end


"""
HELPER FUNCTIONS
"""


""" Compute the norm of each column. """
colnorms(A, p=2) = [norm(A[:, t], p) for t = 1:size(A, 2)]


""" Compute the Euclidean distance between a and b. """
euclid_dist(a, b) = norm(a - b)


""" Cosine of the angle between two vectors. """
cos(a, b) = a'b / (norm(a) * norm(b))


""" Shift cosine angle between h1 and h2. """
function shift_cos(h1, h2, L)
    maxcos = 0
    for l in 0:L-1
        maxcos = max(maxcos, cosL(h1, h2, l))
    end
    return maxcos
end


""" Cosine of the angle between two shifted vectors. """
function cosL(a, b, l, mode="both")
    if (mode == "both")
        return max(cosL(a, b, l, "a"), cosL(a, b, l, "b"))
    elseif (mode == "a")
        return a[1:end-l]'b[1+l:end] / (norm(a[1:end-l]) * norm(b))
    elseif (mode == "b")
        return a[1+l:end]'b[1:end-l] / (norm(a) * norm(b[1:end-l]))
    else
        println("Problem here")
        return nothing
    end
end


""" Normalization matrix. """
function diagscale(c)
    return diagm(0 => c + ones(size(c)) .* (c .< eps()))
end


""" Unit vector. """
function unitvec(n, i)
    e = zeros(n)
    e[i] = 1
    return e
end


""" Return set of maximal values. """
function findsetmax(x; thresh=eps()^(1/2))
    n = length(x)

    maxval = x[1]
    set = [1]

    for i in 2:n
        if (x[i] > maxval + thresh)
            maxval = x[i]
            set = [i]
        elseif (x[i] > maxval - thresh)
            push!(set, i)
        end
    end
            
    return maxval, set
end


function row_normalize(H)
    return inv(diagscale(colnorms(H', 1))) * H
end


"""
EVALUATION TOOLS
"""


function cos_score(trueH, estH)
    K, T = size(trueH)

    score = 0
    for k = 1:K
        score += cos(trueH[k, :], estH[k, :])
    end

    return score / K
end


function permute_factors(trueH, estH)
    # Find permutation by maximizing cosine score
    permset = collect(permutations(1:size(trueH, 1)))     
    resid, i = findmax([cos_score(estH[p, :], trueH) for p in permset])
    return permset[i]
end


""" Check if H contains a diagonal submatrix. """
function is_separable(H, L)
    K, T = size(H)
    
    # Construct block H
    G = zeros(K*L, T)
    for l = 0:L-1
        G[l*K+1:(l+1)*K, l+1:end] = H[:, 1:T-l]
    end

    # Check if G contains the permuted scaled identity
    for r = 1:K*L
        e_r = unitvec(K*L, r)
        found = false

        for t = 1:T
            matches = sum((G[:, t] .!= 0) .== (e_r .!= 0))
            if sum(matches) == K*L
                found = true
                break
            end
        end

        if (!found)
            return false
        else
            println("Found ", r)
        end
    end
    
    return true
end


end  # module
