
using Revise
import JLD
import PyCall
import CMF

scipy_io = PyCall.pyimport("scipy.io")
matfile = scipy_io.loadmat("../../cmf_data/MackeviciusData.mat")
song = matfile["SONG"]

L = 50
K = 3

algorithms = Dict(
    "HALS" => :hals,
    "Mult" => :mult,
    "ANLS" => :anls
)
max_time = 60
seed = sum([Int(c) for c in "INITIALIZE"]);

res = Dict()

# Warmstart algorithms
for alg in keys(algorithms)
    CMF.fit_cnmf(
        song, K=K, L=L,
        alg=algorithms[alg], max_itr=1, max_time=Inf
    )
end
println("Finished warmstart")

for alg in keys(algorithms)
    res[alg] = CMF.fit_cnmf(
        song, K=K, L=L,
        alg=algorithms[alg], max_itr=Inf, max_time=max_time, seed=seed
    )
    
    println("Finished ", alg)
end

JLD.save("songbird.jld", res)