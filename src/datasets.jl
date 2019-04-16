## Manages datasets ##


"""
Generates synthetic data with sparse sequences.

For each unit, the weight across K components is drawn from
a Dirichlet distribution (set `alpha` small for disjoint
sequences).

Parameters
----------
- K : int, number of components.
- N : int, number of units.
- L : int, number of time lags in model.
- T : int, number of time bins in data
- alpha : float, concentation parameter of Dirichlet.
- p_h : float, probability of nonzero entry in H.
- sigma : float, standard deviation of Gaussian.
- noise_scale : float, std of truncated Gaussian noise.

Returns
-------
- data : N × T matrix, holding observed data.
- W : L × N × K tensor, holding ground-truth sequences.
- H : K × T matrix, holding temporal factors.
"""
function synthetic_sequences(;
        K=3,
        N=100,
        L=20,
        T=500,
        alpha=0.1,
        p_h=0.5,
        sigma=0.2,
        noise_scale=1.0,
    )

    # Initialize W with weights of each unit drawn from
    # a Dirichlet distribution.
    mW = transpose(rand(ds.Dirichlet(fill(alpha, K)), N))
    W = repeat(reshape(mW, (1, size(mW)...)), outer=(L, 1, 1))

    # Introduce Gaussian bump with random lag on each component.
    _l = range(-1, stop=1, length=L)
    for (i, j) in Iterators.product(1:N, 1:K)
        cent = rand(ds.Uniform(-1, 1))
        W[:, i, j] .*= ds.pdf.(ds.Gaussian(cent, sigma), _l)
    end

    # Initialize temporal factor with heavy-tailed excursions.
    H = rand(ds.Exponential(), (K, T)) .* rand(ds.Bernoulli(p_h), (K, T))

    # Add noise
    noise = rand(ds.Gaussian(0, noise_scale), (N, T))
    data = max.(0, tensor_conv(W, H) + noise)

    return data, W, H
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
                kernel_width=nothing,
                normalize=false,
                epoch="nothing")
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


"""
Piano dataset.
First 30 seconds of Bach Prelude and Fugue No. 1 in D Major
Stored in a wave file sampled at 44100 Hz

Reference
---------
https://www.youtube.com/watch?v=3srDTD2M8ol
"""
PIANO_DATAPATH = "/home/anthony/cmf_data/prelude_bach.wav"
function piano(;path=PIANO_DATAPATH, freq=11025, seconds=30)
    # Load raw file
    raw, f = WAV.wavread(path)

    # Downsample
    down_rate = Integer(round(f / freq))
    downsampled = raw[1 : down_rate : end]

    # Crop
    last = seconds * freq
    signal = downsampled[1:last]

    # Create spectogram
    spect = DSP.spectrogram(signal, 1024, 512, window=DSP.hamming).power
    
    return spect
end
