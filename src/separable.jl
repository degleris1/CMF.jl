using NonNegLeastSquares
using LinearAlgebra
using PyPlot; plt = PyPlot


include("./common.jl")


function generate_separable_data(;N=23, T=53, K=3, L=5, H_sparsity=0.8, W_sparsity=0.75)
    # Generate W
    trueW = 10 * rand(L, N, K) .* (rand(L, N, K) .> W_sparsity)
    
    # Generate H
    trueH = zeros(K, T)
    for k = 1:K
        trueH[k, (k-1)*L+1] = 1
    end
    trueH[:, K*L+1:end] = rand(K, T-K*L) .* (rand(K, T-K*L) .> H_sparsity)
    
    # Generate data
    X = tensor_conv(trueW, trueH)
    
    return X, trueW, trueH, [N, T, K, L]
end


function fit_conv_separable(data, K, L)
    # Step 1: successive projection to locate the columns of W
    Wo, vertices = SPA(data, K*L)

    # Step 2: compute unconstrained H (NMF)
    Ho = nonneg_lsq(Wo, data, alg=:pivot, variant=:comb)
    
    # Step 3: group and sort rows of H to produce convolutive H
    H, groups = shift_cluster(Ho, K, L)

    # Create W based on grouping
    W = zeros(L, N, K)
    for k = 1:K
        W[:, :, k] = Wo[:, groups[k]]'
    end
    
    return W, H
end


function shift_cluster(Ho, K, L)
    R, T = size(Ho)

    # Step 1: compute distance matrix
    dmat = zeros(R, R)
    for r = 1:R
        for p = r:R
            dmat[r, p] = shift_dist(Ho[r, :], Ho[p, :], L)
            dmat[p, r] = dmat[r, p]
        end
    end
    
    # Step 2: compute groups
    group = [[] for k in 1:K]
    ungrouped = collect(1:L*K)
    for k in 1:K
        # Push a remaining element
        push!(group[k], pop!(ungrouped))

        while (length(group[k]) < L)
            # Add the element closest to the group
            dists = sum(dmat[group[k], ungrouped], dims=1)
            _, i = findmin(dists)
            i = i[2]

            push!(group[k], ungrouped[i])
            deleteat!(ungrouped, i)
        end

        # Sort within group by starting time
        starts = [findfirst(Ho[group[k][i], :] .>= eps()^(1/2)) for i = 1:L]
        group[k] = group[k][sort(collect(1:L), by=j->starts[j])]
    end

    reps = [group[k][1] for k in 1:K]
    return Ho[reps, :], group
end


""" Shift distance between h1 and h2. """
function shift_dist(h1, h2, L; dist=euclid_dist)
    T = length(h1)

    mindist = dist(h1, h2)
    for l in 1:L-1
        mindist = min(mindist,
                      dist(h1[1:T-l], h2[1+l:T]),  # Shift h1 right
                      dist(h2[1+l:T], h2[1:T-l]))  # Shift h2 right
    end

    return mindist
end


""" Successive projection algorithm. """
function SPA(data, K)
    coldata = colnorms(data)
    DX = diagscale(colnorms(data, 1))
    X = data * inv(DX)
    
    vertices = []
    resid = X
    
    for r = 1:K
        _, jset = findsetmax(colnorms(resid))

        if (length(jset) == 1)
            j = jset[1]
        else  # Break ties
            _, k = findmax(coldata[jset])
            j = jset[k]
        end
        push!(vertices, j)

        w = resid[:, j]
        resid = (I - (w * w' / norm(w)^2)) * resid
    end
    
    return data[:, sort(vertices)], sort(vertices)
end


""" Compute the norm of each column. """
colnorms(A, p=2) = [norm(A[:, t], p) for t = 1:size(A, 2)]


""" Normalization matrix. """
function diagscale(c)
    return diagm(0 => c + ones(size(c)) .* (c .< eps()))
end


""" Compute the Euclidean distance between a and b. """
euclid_dist(a, b) = norm(a - b)


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



"""
Test
"""
data, tW, tH, (N, T, K, L) = generate_separable_data()

W, H = fit_conv_separable(data, K, L)



plt.figure()
plt.imshow(H, aspect="auto")
plt.title("Fit")
plt.show()

plt.figure()
plt.imshow(tH, aspect="auto")
plt.title("Truth")
plt.show()
