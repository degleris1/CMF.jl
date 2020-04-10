using Combinatorics
using LinearAlgebra
using PyPlot

""" Visualize Ws """
function vizW(estW, trueW=nothing)
    if isnothing(trueW)
        nrows = 1
    else
        nrows = 2
    end

    K, N, L = size(estW)

    for k = 1:K
        plt.subplot(nrows, K, k)
        plt.imshow(estW[k, :, :], aspect="auto")
        plt.xticks([])
        plt.yticks([])

        k == 1 && plt.ylabel("Estimate")
    end
    if !isnothing(trueW)
        for k = 1:K
            plt.subplot(nrows, K, K+k)
            plt.imshow(trueW[k, :, :], aspect="auto")
            plt.xticks([])
            plt.yticks([])

            k == 1 && plt.ylabel("Truth")
        end
    end
end

""" Evaluate recovered kernels `estW` against true kernels `trueW`. """
function evalW(estW, trueW)
    K, N, L = size(estW)
    Kt, Nt, Lt = size(trueW)
    @assert (N == Nt) && (K == Kt) && (L >= Lt)        

    bestscore = Inf
    bestperm = nothing
    bestlags = nothing

    # Compare across all permutations and shifts
    permset = collect(permutations(1:K))
    lagset = collect(0:L-Lt)

    for perm in permset
        score = 0
        lags = []
        for k = 1:K
             tw = trueW[k, :, :]

             ewlist = [estW[perm[k], :, l+1:l+Lt] for l in lagset]
             scores = [norm(ew/norm(ew) - tw/norm(tw))^2 for ew in ewlist]
             err, ind = findmin(scores)
             score += err
             push!(lags, lagset[ind])
        end

        if score < bestscore
            bestscore = score
            bestperm = perm
            bestlags = lags
        end           
    end

    # Generate tuned estimate
    bestest = zeros(K, N, Lt)
    for k = 1:K
        bestest[k, :, :] = estW[bestperm[k], :, bestlags[k]+1:bestlags[k]+Lt]
    end

    return bestscore, bestperm, bestlags, bestest
end