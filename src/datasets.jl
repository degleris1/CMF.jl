using HDF5
using ImageFiltering

function gen_synthetic(;K=3, N=100, L=20, T=50, H_sparsity=0.9, noise_scale=1.0)
    # Generate random convolutional parameters
    W = zeros(L, N, K)
    H = rand(K, T)

    # Add sparsity
    H = H .* (rand(K, T) .> H_sparsity) 
    
    # Add structure
    for i = 1:N
        j = rand(1:K)
        W[:, i, j] += _gauss_plus_delay(L)
    end

    # Add noise
    noise = noise_scale * rand(N, T)
    data = tensor_conv(W, H) + noise

    return data
end


function _gauss_plus_delay(n_steps)
    tau = rand() * 3 - 1.5
    x = range(-3-tau, stop=3-tau, length=n_steps)
    y = exp.(-x.^2)
    return y / maximum(y)
end

"""
Silicon-Probe neural recordings from rats before, during, and after
a maze-running task.
Reference
---------
https://crcns.org/data-sets/hc/hc-11
"""
MAZE_DATAPATH = "~/cmf_data/NoveltySessInfoMatFiles/Achilles_10252013_sessInfo.mat"
function maze(;path=MAZE_DATAPATH,
                start_time=0,
                end_time=200,
                bin_time=1e-1,
                kernel_width=nothing)
    f = h5open(path, "r") do file
        read(file, "sessInfo/Spikes")
    end

    spike_ids = f["SpikeIDs"]
    spike_times = f["SpikeTimes"]

    # Only a few neurons have spikes in our data, so we remove
    # unneeded neurons by forming a map from spike_id -> neuron
    id_map = Dict()
    for (i, neuron) in enumerate(unique(spike_ids))
        id_map[neuron] = i
    end
    neuron_assignments = [id_map[x] for x in spike_ids]

    # Reject spikes outside of our time window
    # An end time of -1 corresponds to using all data
    if (end_time == -1)
        end_time = spike_ids[-1]
    end

    spike_idx = (spike_times .>= start_time) .& (spike_times .<= end_time)
    neuron_assignments = neuron_assignments[spike_idx]
    spike_times = spike_times[spike_idx]

    num_bins = Int((end_time - start_time) / bin_time) + 1
    num_neurons = maximum(neuron_assignments)
    data = zeros(num_neurons, num_bins)
    spike_times_binned = Int.(round.((spike_times .- start_time) ./ (bin_time)) .+ 1)

    # Set the data matrix by iterating over each spike, and incrementing
    # its time bin.
    num_spikes = size(neuron_assignments, 1)
    for n in 1:num_spikes
        neuron = neuron_assignments[n]
        bin = spike_times_binned[n]
        data[neuron, bin] += 1
    end

    # If kernel_width is passed, convolve each row with a gaussian kernel
    # of the given width. The width specifies the standard deviation of the kernel.
    if kernel_width != nothing
        kern = KernelFactors.gaussian((0,kernel_width))
        data = imfilter(data,kern)
    end

end
;
