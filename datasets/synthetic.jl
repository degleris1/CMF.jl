"""
Generates synthetic data with sparse sequences.

For each unit, the weight across K components is drawn from
a Dirichlet distribution (set `alpha` small for disjoint
sequences).

Parameters
----------
- K : int, number of components.
- N : int, number of units.
- L : int, number of time lags in model.
- T : int, number of time bins in data
- alpha : float, concentation parameter of Dirichlet.
- p_h : float, probability of nonzero entry in H.
- sigma : float, standard deviation of Gaussian.
- noise_scale : float, std of truncated Gaussian noise.

Returns
-------
- data : N × T matrix, holding observed data.
- W : L × N × K tensor, holding ground-truth sequences.
- H : K × T matrix, holding temporal factors.
"""
function synthetic_sequences(;
        K=3,
        N=100,
        L=20,
        T=500,
        alpha=0.1,
        p_h=0.5,
        sigma=0.2,
        noise_scale=1.0,
    )

    # Initialize W with weights of each unit drawn from
    # a Dirichlet distribution.
    mW = transpose(rand(ds.Dirichlet(fill(alpha, K)), N))
    W = repeat(reshape(mW, (1, size(mW)...)), outer=(L, 1, 1))

    # Introduce Gaussian bump with random lag on each component.
    _l = range(-1, stop=1, length=L)
    for (i, j) in Iterators.product(1:N, 1:K)
        cent = rand(ds.Uniform(-1, 1))
        W[:, i, j] .*= ds.pdf.(ds.Gaussian(cent, sigma), _l)
    end

    # Initialize temporal factor with heavy-tailed excursions.
    H = rand(ds.Exponential(), (K, T)) .* rand(ds.Bernoulli(p_h), (K, T))

    # Add noise
    noise = rand(ds.Gaussian(0, noise_scale), (N, T))
    data = max.(0, tensor_conv(W, H) + noise)

    return data, W, H
end
