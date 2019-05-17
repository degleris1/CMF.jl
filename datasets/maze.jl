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
                kernel_width=nothing,
                normalize=false,
                epoch=nothing)
    """
    params

        path:         Path to to a sessInfo.mat file from the CRCNS dataset
        start_time:   Earliest time for which we retrieve data. If "epoch" is specified
                      this is the time *after* the beginning of that epoch for which we begin
                      looking for spikes.
        end_time:     Latest time for which we consider spikes. If epoch is specified, this must
                      be less than the duration of the epoch, otherwise we raise an error.
        bin_time:     The length of each bin, in seconds.
        kernel_width: The standard deviation of the guassian kernel for smoothing. If nothing,
                      we don't do any smoothing.
        normalize:    If true, normalize each row using the sum of absolute values.
        epoch:        The name of the epoch for which to retrieve data. Can be one of:
                      "PRE", "MAZE", or "POST". Otherwise gives an error.
    """

    f = h5open(path, "r") do file
        read(file, "sessInfo/Spikes")
    end

    g = h5open(path, "r") do file
        read(file, "sessInfo/Epochs")
    end

    spike_ids = f["SpikeIDs"]
    spike_times = f["SpikeTimes"]

    # Reject spikes outside of our time window
    # An end time of -1 corresponds to using all data
    if epoch != nothing
        if epoch == "PRE"
            epoch_start, epoch_stop = g["PREEpoch"]
        elseif epoch == "MAZE"
            epoch_start, epoch_stop = g["MazeEpoch"]
        elseif epoch == "POST"
            epoch_start, epoch_stop = g["POSTEpoch"]
        else
            error("Invalid epoch name given.")
        end

        start_time = epoch_start + start_time

        if (end_time == -1)
            end_time = epoch_stop
        else
            end_time = epoch_start + end_time
        end

        # Ensure start and end times are valid
        @assert(start_time >= epoch_start)
        @assert(end_time <= epoch_stop)
    else
        if (end_time == -1)
            end_time = spike_ids[end]
        end
    end

    # Only a few neurons have spikes in our data, so we remove
    # unneeded neurons by forming a map from spike_id -> neuron
    id_map = Dict()
    for (i, neuron) in enumerate(unique(spike_ids))
        id_map[neuron] = i
    end
    neuron_assignments = [id_map[x] for x in spike_ids]

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

    # optionally zscore each neuron
    if normalize
        for i in 1:num_neurons
            data[i,:] = data[i,:] ./ sum(abs.(data[i,:]))
        end
    end

    return data
end
