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

algs = [
    Dict(:name => "mult",
         :opts => Dict()
    ),
    Dict(:name => "hals",
         :opts => Dict()
    ),
    Dict(:name => "annls",
         :opts => Dict()
    )
]

all_results = Dict()
for (dataset_name, data) in datasets
    all_results[dataset_name] = Dict()

    for alg in algs
        alg_name = alg[:name]

        results = fit_cnmf(data, 
                            L=20, K=3,
                            alg=alg_name, alg_options=alg[:opts],
                            max_itr=200, max_time=6)

        all_results[dataset_name][alg_name] = results
    end
end

JLD.save(RESULTS_PATH, "all_results", all_results)

;