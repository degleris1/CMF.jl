using HDF5

include("../src/model.jl")
include("../src/datasets.jl")
include("../src/visualize.jl")

RESULTS_PATH = "alg_comparison_results.h5"

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

fid = h5open(RESULTS_PATH, "w")
all_results = Dict()
for (dataset_name, data) in datasets
    all_results[dataset_name] = Dict()

    for (alg, options) in algs
        results = fit_cnmf(data, 
                            L=20, K=3,
                            alg=alg, alg_options=options,
                            max_itr=200, max_time=30)

        # Store results from each algorithm/dataset combination using
        # a separate HDF5 group, to allow read/write from Python.
        g = HDF5.g_create(fid, "results/$dataset_name/$alg")
        g["W"] = results.W
        g["H"] = results.H
        g["data"] = results.data
        g["loss_hist"] = results.loss_hist
        g["time_hist"] = results.time_hist
    end
end

close(fid)
;
