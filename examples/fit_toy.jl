using Revise
using CMF
using Random

import PyPlot
const plt = PyPlot

include("../datasets/toy.jl")

function example_toy()
    Random.seed!(4)
    X, W, H = generate_toy_data()
    maxlag = size(W, 1)
    K = size(W, 3)

    results = fit_cnmf(X, L=maxlag*3, K=K, alg=CMF.HALSUpdate, max_itr=300, l1_W=10, l1_H=1)

    println("Loss: ", results.loss_hist[end])
    CMF.plot_reconstruction(results, 1:50, sort_units=false)
    CMF.plot_Ws(results, sort_units=false)
    plt.show()
end
example_toy()