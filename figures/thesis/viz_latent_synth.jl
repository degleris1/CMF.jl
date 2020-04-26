using MAT
using PyPlot

using Revise
using CMF

include("eval.jl")
include("plot_settings.jl")

plt.close()

# Load results
folder = "/home/asd/data/thesis/"
ntrials = 20
num_noise = 20
noiselevels = exp.( range(log(0.1), log(10), length=num_noise) )

results = Dict()

for noise in 1:num_noise
    results[noise] = Dict()
    for trial = 1:ntrials
        results[noise][trial] = matread(
            string(folder, "latsyn/", noise, "_", trial, ".mat")
        )
    end
end


# Qualitative result: plot recovered motifs for two noise levels
# trial = 5 
# W = results[1][trial]["trueW"]
# Wl = results[1][trial]["estW"]  # Low noise
# Wh = results[10][trial]["estW"]  # High noise

# fig1 = plt.figure(figsize=set_size())
# score, perm, offsets, estl = evalW(Wl, W)
# score, perm, offsets, esth = evalW(Wh, W)
# K, N, L = size(Wl)
# for (i, mat) in enumerate([W, estl, esth])
#     for k = 1:K
#         ind = (i-1)*K+k
#         plt.subplot(3, K, ind)
#         plt.imshow(mat[k, :, :], aspect="auto")
#         plt.xticks([])
#         plt.yticks([])

#         ind == 1 && plt.ylabel("Truth")
#         ind == 6 && plt.ylabel("\$\\sigma = 0.1\$")
#         ind == 11 && plt.ylabel("\$\\sigma = 10\$")
#     end
# end
# plt.tight_layout()
# plt.savefig(folder * "latsyn_kernels.eps")

# Quantitative result: plot (log) recovery score against noise level
scores = zeros(length(noiselevels), ntrials)
for (noise, trial) in Iterators.product(1:num_noise, 1:ntrials)
    r = results[noise][trial]
    score, perm, offsets, est = evalW(r["estW"], r["trueW"], )
    scores[noise, trial] = score
end
avgscores = sum(scores, dims=2) / ntrials

fig3 = plt.figure(figsize=set_size(aspect=0.6*golden_ratio))
plt.plot(noiselevels, avgscores)
plt.ylabel("Error")
plt.xlabel("Noise Level \$ \\sigma \$")
plt.xscale("log")
plt.xlim(0.1, 10)
plt.tight_layout()
plt.savefig(folder * "latsyn_error.eps")

