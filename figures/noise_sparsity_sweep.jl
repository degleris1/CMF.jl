using Plots
import JLD

include("../src/model.jl")
include("../src/datasets.jl")


# Algorithm settings
algs_under_test = [
    Dict(:name => "mult",
         :opts => Dict(),
         :label => "MULT"),
    Dict(:name => "hals",
         :opts => Dict(),
         :label => "HALS")
]

max_itr = 100_000
max_time = 20


# Data settings
N, T = 100, 500
K, L = 3, 10

noise_levels = 0 : 0.2 : 1  # 0 : 0.1 : 1
sparsity_levels = [0.99]  # [0, 0.5, 0.9, 0.99] 


# Run tests
all_results = Dict()
for noise in noise_levels
    all_results[noise] = Dict()

    for sparsity in sparsity_levels
        all_results[noise][sparsity] = Dict()
        
        println("Testing noise ", noise, " and sparsity ", sparsity)
        # Generate data
        data = gen_synthetic(N=N, T=T, L=L, K=K, H_sparsity=sparsity, noise_scale=noise)

        # Test algorithms
        for alg in algs_under_test
            results = fit_cnmf(data, L=L, K=K,
                               alg=alg[:name], alg_options=alg[:opts],
                               max_itr=max_itr, max_time=max_time)

            all_results[noise][sparsity][alg[:name]] = results
            println(alg[:label], ": ", results.loss_hist[end]) 
        end
    end
end

# Save
JLD.save("./noise_sparsity_sweep.jld", "all_results", all_results)

# Generate plot data
plot(xlabel="Noise scale", ylabel="Loss")
for alg in algs_under_test
    perf = [all_results[noise][0.99][alg[:name]].loss_hist[end]
            for noise in noise_levels]
    plot!(noise_levels, perf, label=alg[:name])
end        
gui()
