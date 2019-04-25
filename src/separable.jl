using NonNegLeastSquares
using LinearAlgebra
using PyPlot; plt = PyPlot


include("./common.jl")


function generate_separable_data(;N=100, T=250, K=3, L=8, H_sparsity=0.15, W_sparsity=0.75)
    # Generate W
    trueW = 100 * rand(L, N, K) .* (rand(L, N, K) .> W_sparsity)

    # Generate H
    trueH = zeros(K, T)
    for k = 1:K
        trueH[k, (k-1)*L+1] = 1
    end
    trueH[:, K*L+1:end] = rand(K, T-K*L) .* (rand(K, T-K*L) .> H_sparsity)
    trueH[:, T-L+1:end] .= 0
    
    # Generate data
    X = tensor_conv(trueW, trueH)
    
    return X, trueW, trueH, [N, T, K, L]
end


function fit_separable_data(data, K, L)
    # Step 0: normalize columns of X
    DX = diagm(0 => max.(colnorms(data), eps()))
    X = data * inv(DX)

    # Step 1: successive projection to locate the columns of W
    Wo, vertices = SPA(X, K*L)

    # Step 1B: renormalize W
    DW = DX[vertices, vertices]
    Wo = Wo * DW
    
    # Step 2: compute unconstrained H (NMF)
    Ho = nonneg_lsq(Wo, data, alg=:pivot, variant=:comb)

    # Step 3: group rows of H to produce convolutive H
    H, groups = shift_cluster(Ho, K, L)

    # Step 4: create W based on grouping
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
    for r = 1:R
        for p = r:R
            simat[r, p] = compute_sim(Ho[:, r], Ho[:, p], L)
            simat[p, r] = simat[r, p]
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
            sims = sum(simat[group[k], ungrouped], dims=1)
            _, i = findmax(sims)
            i = i[2]

            push!(group[k], ungrouped[i])
            deleteat!(ungrouped, i)
        end

        # Sort within group by starting time
        starts = [findfirst(Ho[group[k][i], :] .>= eps()^(1/2)) for i = 1:L]
        println(group[k], starts)
        group[k] = group[k][sort(collect(1:L), by=j->starts[j])]
    end

    
    reps = [group[k][1] for k in 1:K]
    
    return Ho[reps, :], group
end




function compute_sim(h1, h2, L)
    T = length(h1)
    
    best_sim = abs(angle(h1, h2))

    # Shift h1 or h2 right
    for l = 1:L-1
        best_sim = max(
            best_sim,
            abs(angle(h1[1:T-l], h2[1+l:T])),  # Shift h1 right
            abs(angle(h1[1+l:T], h2[1:T-l]))
        )
    end
    
    return best_sim
end



function SPA(X, n)
    vertices = []
    R = X
    
    for r = 1:n
        _, j = findmax(colnorms(R))
        push!(vertices, j)

        w = R[:, j]
        R = (I - (w * w' / norm(w)^2)) * R
    end

    return X[:, sort(vertices)], sort(vertices)
end

    
colnorms(A) = [norm(A[:, t]) for t = 1:size(A, 2)]
angle(a, b) = a'b / (norm(a) * norm(b))
    

data, tW, tH, (N, T, K, L) = generate_separable_data()

W, H = fit_separable_data(data, K, L)

plt.figure()
plt.imshow(H, aspect="auto")
plt.title("Fit")
plt.show()

plt.figure()
plt.imshow(tH, aspect="auto")
plt.title("Truth")
plt.show()
