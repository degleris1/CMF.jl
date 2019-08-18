
import CMF
import PyPlot; plt = PyPlot;
import Random; Random.seed!(0);
import JLD
using ArgParse
using Dates
using Random
include("../datasets/synthetic.jl")


# Parse commandline arguments to allow specifying model size
s = ArgParseSettings()
@add_arg_table s begin
    "--N"
        help = "Number of features"
        arg_type = Int
        default = 250
    "--K"
        help = "Number of components"
        arg_type = Int
        default = 5
    "--L"
        help = "Lag"
        arg_type = Int
        default = 20
    "--alpha"
        help = "Dirichlet param alpha for generating data"
        arg_type = Float64
        default = 0.1
    "--sigma"
        help = "Dirichlet param sigma for generating data"
        arg_type = Float64
        default = 0.2
    "--p_h"
        help = "Probability of nonzero h"
        arg_type = Float64
        default = 0.1
    "--noise_scale"
        help = "Noise scale for synthetic data"
        arg_type = Float64
        default = 0.1
end
parsed_args = parse_args(ARGS, s)


# Model params
N = parsed_args["N"]
K = parsed_args["K"]
L = parsed_args["L"]

# Synthetic data params
alpha = parsed_args["alpha"]
p_h = parsed_args["p_h"]
sigma = parsed_args["sigma"]
noise_scale = parsed_args["noise_scale"]

T_list = [500, 2500, 10_000, 50_000]
runtimes = Dict(
    500 => 60,
    2500 => 120,
    10_000 => 400,
    50_000 => 1000
)

alg_list = [:hals, :mult, :anls]
labels = Dict(:hals => "HALS", :mult => "MULT", :anls => "ANLS")

results = Dict()
results["args"] = parsed_args

data_list = Dict()

for T in T_list
    data, trueW, trueH = synthetic_sequences(
        K=K, N=N, L=L, T=T,
        alpha=alpha, 
        p_h=p_h, 
        sigma=sigma, 
        noise_scale=noise_scale
    )
    data_list[T] = (data, trueW, trueH)
end

# Run once to warmup algorithms
for T in [500]
    for alg in alg_list
        CMF.fit_cnmf(data_list[T][1], alg=alg, max_itr=1)
    end
end

for T in T_list
    results[T] = Dict()
    
    for alg in alg_list
        # use same initialization for all algs
        Random.seed!(0)

        results[T][alg] = CMF.fit_cnmf(
            data_list[T][1], alg=alg,
            K=K, L=L, max_time=runtimes[T], max_itr=Inf
        )
    end
end

time_now = string(now())
JLD.save("./synthetic_comparison_$time_now.jld", "results", results)

