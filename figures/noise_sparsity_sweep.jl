using Plots
import JLD

include("../src/model.jl")
include("../src/datasets.jl")
include("../src/common.jl")


# Algorithm settings
algs_under_test = [
    Dict(:name => :mult,
         :opts => Dict(),
         :label => "MULT"),
    Dict(:name => :hals,
         :opts => Dict(),
         :label => "HALS"),
    Dict(:name => :anls,
         :opts => Dict(),
         :label => "ANLS"),
]

max_itr = 100_000
max_time = 60


# Data settings
N, T = 100, 500
K, L = 3, 10

noise_levels = 0.1 : 0.1 : 1
sparsity_levels =  [0.5, 0.6, 0.7, 0.8, 0.85,
                    0.9, 0.95, 0.97, 0.99, 0.999] 


# Run tests
all_results = Dict()
for noise in noise_levels
    all_results[noise] = Dict()

    for sparsity in sparsity_levels
        all_results[noise][sparsity] = Dict()
        
        println("Testing noise ", noise, " and sparsity ", sparsity)
        # Generate data
        data, truth = gen_synthetic(N=N, T=T, L=L, K=K,
                                    H_sparsity=sparsity, noise_scale=noise,
                                    return_truth=true)

        JLD.save(string("./sweep/data_", noise, "_", sparsity, ".jld"), "truth", truth) 
        
        # Test algorithms
        for alg in algs_under_test
            results = fit_cnmf(data, L=L, K=K,
                               alg=alg[:name], alg_options=alg[:opts],
                               max_itr=max_itr, max_time=max_time)

            # Save algorithm results
            save_model(results, string("./sweep/",
                                       alg[:label], "_",
                                       noise, "_",
                                       sparsity, ".h5")
                       )
            
            all_results[noise][sparsity][alg[:name]] = results
            println(alg[:label], ": ", compute_loss(truth, results.W, results.H)) 
        end
    end
end

# Save
JLD.save("./noise_sparsity_sweep.jld", "all_results", all_results)
