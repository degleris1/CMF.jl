using NonNegLeastSquares
using LinearAlgebra
using PyPlot; plt = PyPlot


include("./common.jl")


function generate_separable_data(;N=100, T=250, K=3, L=8, H_sparsity=0.8, W_sparsity=0.75)
    # Generate W
    trueW = 10 * rand(L, N, K) .* (rand(L, N, K) .> W_sparsity)
    
    # Generate H
    trueH = zeros(K, T)
    for k = 1:K
        trueH[k, (k-1)*L+1] = 1
    end
    trueH[:, K*L+1:end] = rand(K, T-K*L) .* (rand(K, T-K*L) .> H_sparsity)
    # Make entire
    trueH[:, T-L:end] .= 0
    # Columns should sum to 1
    trueH = trueH * inv(diagnorm(trueH, 1))
    
    # Generate data
    X = tensor_conv(trueW, trueH)
    
    return X, trueW, trueH, [N, T, K, L]
end


function fit_conv_separable(data, K, L)
    # Step 1: successive projection to locate the columns of W
    Wo, vertices = SPA(data, K*L)

    # Step 2: compute unconstrained H (NMF)
    Ho = nonneg_lsq(Wo, data, alg=:pivot, variant=:comb)
    
    # DEBUG
    println("NNLS loss: ", norm(Wo * Ho - data))
    println(vertices)
    # -----
    
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

    # Step 1: compute similarity matrix
    simat = zeros(R, R)
    dmat = zeros(R, R)
    for r = 1:R
        for p = r:R
            simat[r, p] = compute_sim(Ho[r, :], Ho[p, :], L)
            simat[p, r] = simat[r, p]

            dmat[r, p] = shift_dist(Ho[r, :], Ho[p, :], L)
            dmat[p, r] = dmat[r, p]
        end
    end

    plt.figure()
    plt.title("Similarity")
    plt.imshow(simat)
    plt.colorbar()

    plt.figure()
    plt.title("Distance")
    plt.imshow(dmat)
    plt.colorbar()
    
    # Step 2: compute groups
    group = [[] for k in 1:K]
    ungrouped = collect(1:L*K)
    for k in 1:K
        # Push a remaining element
        push!(group[k], pop!(ungrouped))

        while (length(group[k]) < L)
            # Add the element closest to the group
            sims = sum(simat[group[k], ungrouped], dims=1)
            _, i = findmax(sims)
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


""" Shift similarity between h1 and h2. """
function compute_sim(h1, h2, L)
    T = length(h1)
    
    best_sim = cosine(h1, h2)
    for l = 1:L-1
        best_sim = max(best_sim,
                       cosine(h1[1:T-l], h2[1+l:T]),  # Shift h1 right
                       cosine(h1[1+l:T], h2[1:T-l]))
    end
    
    return best_sim
end


""" Successive projection algorithm. """
function SPA(data, n)
    DX = diagnorm(data, 1)
    X = data * inv(DX)
    
    vertices = []
    R = X
    
    for r = 1:n
        _, j = findmax(colnorms(R))
        push!(vertices, j)

        w = R[:, j]
        R = (I - (w * w' / norm(w)^2)) * R
    end
    
    return data[:, sort(vertices)], sort(vertices)
end


""" Compute the norm of each column. """
colnorms(A, p=2) = [norm(A[:, t], p) for t = 1:size(A, 2)]


""" Normalization matrix. """
function diagnorm(A, p=2)
    c = colnorms(A, p)
    return diagm(0 => c + ones(size(c)) .* (c .< eps()))
end


""" Compute the cosine of the angle between two vectors. """
cosine(a, b) = a'b / (norm(a) * norm(b))


""" Compute the Euclidean distance between a and b. """
euclid_dist(a, b) = norm(a - b)


"""
Test
"""
data, tW, tH, (N, T, K, L) = generate_separable_data()

W, H = fit_conv_separable(data, K, L)


"""
plt.figure()
plt.imshow(H, aspect="auto")
plt.title("Fit")
plt.show()

plt.figure()
plt.imshow(tH, aspect="auto")
plt.title("Truth")
plt.show()
"""
