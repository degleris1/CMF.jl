using Revise
using CMF
using Random

import CMF: tensor_conv

import PyPlot
const plt = PyPlot

include("../datasets/waveform.jl")


function example_trianglewave()
    Random.seed!(1234)
    x = generate_trianglewave()
    x = reshape(x, (1, length(x)))

    L = 50
    K = 1

    res = fit_cnmf(
        x; L=50, K=1, 
        max_itr=300, alg=HALSUpdate, l1_H=0.05, initH=copy(x)
    )

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title("Truth")
    plt.plot(x')
    plt.subplot(3, 1, 2)
    plt.title("Reconstruction")
    plt.plot(tensor_conv(res.W, res.H)')
    plt.subplot(3, 1, 3)
    plt.title("H")
    plt.plot(res.H[1, :]', color="r")
    plt.show()

    plt.figure()
    plt.plot(res.W[:, 1, 1]')
    plt.show()
end
example_trianglewave()