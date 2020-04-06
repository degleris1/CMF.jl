import CMF
import PyPlot
const plt = PyPlot

function generate_toy_data()
    N, T = 7, 50
    K, L = 2, 5

    # Construct motifs
    W = zeros(L, N, K)
    W[:, :, 1] = [
        0 0 0 0 0;
        0 0 0 0 0;
        0 4 0 4 0;
        0 0 0 0 0;
        3 0 0 0 3;
        0 2 1 2 0;
        0 0 0 0 0
    ]'
    W[:, :, 2] = [
        0 1 0 0 0;
        0 0 2 0 0;
        0 0 0 3 0;
        0 5 0 0 0;
        0 0 5 0 0;
        0 0 0 5 0;
        0 0 0 0 0;
    ]'

    # Construct activation pattern
    H = zeros(K, T)

    H[1, 2] = 1
    H[1, 20] = 2
    H[1, 32] = 1
    H[1, 48] = 1


    H[2, 12] = 1
    H[2, 30] = 1
    H[2, 38] = 0.5

    H = [H H H H H]

    X = CMF.tensor_conv(W, H)

    return X, W, H
end

function display_toy_data()
    X, W, H = generate_toy_data()
    plt.figure()
    plt.imshow(X, cmap="viridis")
end