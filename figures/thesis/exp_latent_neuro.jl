using Random
using MAT

using Revise
using CMF

include("eval.jl")
include("plot_settings.jl")

folder = "/home/asd/data/thesis/"
Random.seed!(7)

# Load data
matdict = matread(folder * "mackdata.mat")
data = matdict["NEURAL"]
perm = [0, 12, 21, 24, 28, 29, 39, 46, 70, 72, 74, 14, 65,  3, 36, 57, 45,
        10,  2, 26, 40, 54, 50, 62,  9, 37, 63, 35, 66,  5, 32, 38, 41, 68,
        69, 61, 16, 11, 56, 33, 55, 60,  4, 18, 19, 31, 27, 30, 42, 23, 47,
        48, 67, 17, 43, 44, 52, 53, 71, 13, 22, 51,  7,  8, 59,  6, 15,  1,
        73, 64, 49, 20, 25, 34, 58] .+ 1
data = data[perm, :]

N, T = size(data)
K = 2
L = 50

#l1range = exp.(range(log(0.1), log(1.5), length=20))
#for λ in l1range

# Good parameters
# W abs 0.11, H abs 1.3, sq loss
# W sq 0.01, H abs 1, sq los

# Fit data
r = fit_cnmf(
    data; L=L, K=K,
    alg=PGDUpdate, 
    max_itr=Inf, 
    max_time=20, 
    tol=1e-4,
    penaltiesW=[CMF.SquarePenalty(0.01)],  # 0.11
    penaltiesH=[CMF.AbsolutePenalty(1)],  # 1.3
    loss_func=CMF.SquareLoss(),
)

# Normalize factors
for k = 1:K
    mag = norm(r.W[k, :, :])
    r.W[k, :, :] /= mag
    r.H[k, :] *= mag
end


# plt.figure()
# plt.plot(l1range, losses)
# plt.xscale("log")

plt.close()
plt.figure()
vizW(r.W)

# Generate plot
plt.close()
fig = plt.figure(figsize=set_size())
nrows = 10
ncols = 10
offset = 1
shape = (nrows, ncols)
colors = ["r", "b"]

# Plot observed data
plt.subplot2grid(shape, (2, 2), rowspan=nrows-2, colspan=ncols-2)
plt.yticks([])
plt.xticks([])
plt.imshow(data, aspect="auto")
plt.xlabel("Timebin \$t\$")

# Plot motifs
locs = [1, 0]
for k = 1:2
    plt.subplot2grid(shape, (2, locs[k]), rowspan=nrows-2, colspan=1)
    plt.imshow(r.W[k, :, :], aspect="auto")

    k == 2 && plt.yticks([1, 25, 50, 75] .- 1, [1, 25, 50, 75])
    k == 1 && plt.yticks([])
    plt.xticks([])
    plt.xlabel("\$w_{n" * string(k) * "}\$")
    k == 2 && plt.ylabel("Neuron \$n\$")
end

# Plot feature maps
offset = ceil(Int, maximum(r.H)) + 0.5
plt.subplot2grid(shape, (0, 2), rowspan=2, colspan=ncols-2)
plt.xlim(1, T)
plt.xticks([])
plt.yticks([1offset, 2offset], ["\$h_{1}\$", "\$h_{2}\$"])
plt.ylim(0.5offset, 3offset)
for k = 1:2
    plt.plot(1:T, r.H[k, :] .+ offset * k, color=colors[k])
end
#plt.tight_layout()
plt.savefig(folder * "latneuro.eps")