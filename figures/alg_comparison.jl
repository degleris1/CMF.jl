using JLD

include("../src/model.jl")
include("../src/datasets.jl")
include("../src/visualize.jl")

RESULTS_PATH = "alg_comparison_results.jld"

# Initialize dictionaries to store datasets
datasets = Dict(
    :synthetic => gen_synthetic(N=500, T=2000),
    :maze => maze(path="../NoveltySessInfoMatFiles/Achilles_10252013_sessInfo.mat",
                  bin_time=1e-1,
                  kernel_width=2)
)

# alg -> options
algs = Dict(
    :mult => Dict(),
    :hals => Dict(),
    :anls => Dict()
) 

all_results = Dict()
for (dataset_name, data) in datasets
    all_results[dataset_name] = Dict()

    for (alg, options) in algs
        results = fit_cnmf(data, 
                            L=20, K=3,
                            alg=alg, alg_options=options,
                            max_itr=200, max_time=30)

        all_results[dataset_name][alg] = results
    end
end

JLD.save(RESULTS_PATH, "all_results", all_results)

;