using Plots
using LaTeXStrings

include("../src/model.jl")
include("../src/datasets.jl")

# Initialize dictionaries to store datasets
synthetic_dset = Dict(
                        :data => gen_synthetic(N=500, T=2000),
                        :name => "synthetic")

maze_dset = Dict(
                    :data => maze(path="../../NoveltySessInfoMatFiles/Achilles_10252013_sessInfo.mat"),
                    :name => "maze")

datasets = [synthetic_dset, maze_dset]

# Initialize dictionaries to store algorithms
mult_alg = Dict(
                :name => "mult",
                :alg_opts => Dict(),
                :label => "MULT",
                :average_loss_hist => nothing,
                :average_time_hist => nothing
)

hals_alg = Dict(
                :name => "hals",
                :alg_opts => Dict("mode" => "elementwise"),
                :label => "EW-HALS",
                :average_loss_hist => nothing,
                :average_time_hist => nothing
)
algs = [mult_alg, hals_alg]

NUM_TRIALS = 10

plot(xlabel="Time", ylabel="Loss")

for dataset in datasets
    data = dataset[:data]
    dataset_name = dataset[:name]

    # Create a new figure for each dataset
    p = plot()

    # Average over multiple trials to account for differences
    # due to random initialization.
    for alg in algs
        alg[:average_loss_hist] = nothing
        alg[:average_time_hist] = nothing
        for trial in 1:NUM_TRIALS

            results = fit_cnmf(data, 
                                L=10, K=5,
                                alg=alg[:name], alg_options=alg[:alg_opts],
                                max_itr=Inf, max_time=30)

            time_hist = results.time_hist
            loss_hist = results.loss_hist

            # Update loss and time hists, keeping a running
            # average of each. Since the length of the arrays can vary from
            # trial to trial, we only keep the indices corresponding to the shortest arrays.
            if alg[:average_loss_hist] == nothing
                alg[:average_loss_hist] = loss_hist
            else
                t = min(length(alg[:average_loss_hist]), length(loss_hist))
                alg[:average_loss_hist] = alg[:average_loss_hist][1:t]
                alg[:average_loss_hist] += loss_hist[1:t]
            end

            if alg[:average_time_hist] == nothing
                alg[:average_time_hist] = time_hist
            else
                t = min(length(alg[:average_time_hist]), length(time_hist))
                alg[:average_time_hist] = alg[:average_time_hist][1:t]
                alg[:average_time_hist] += time_hist[1:t]
            end

        end

        alg[:average_loss_hist] ./= NUM_TRIALS
        alg[:average_time_hist] ./= NUM_TRIALS

        loss_hist = alg[:average_loss_hist]
        time_hist = alg[:average_time_hist]

        t = min(length(time_hist), length(loss_hist))
        plot!(time_hist[1:t], loss_hist[1:t], label=alg[:label])
        scatter!(time_hist[1:t], loss_hist[1:t], markersize=1, label="")
        ylabel!(L"\frac{||X - \tilde{X}||}{||X||}")
        xlabel!(L"\textnormal{Time (s)}")
        savefig("cnmf_alg_comparison_$dataset_name.png")
    end
end
    
gui()
;
