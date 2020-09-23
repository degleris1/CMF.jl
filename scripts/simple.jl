using CMF
using MLJBase

# Generate data
N, T = 30, 100
X = rand(N, T)

model = ConvolutionalFactorization()
fit(model, 0, X)