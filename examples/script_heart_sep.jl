using MAT
using CMF

# Desired settings
# K = 10, L = 30

function runscript(filename, K, L, beta)
    path = "/home/asd/data/heart/"
    md = matread(path * "train/" * filename * ".mat")
    signal = md["data"]

    N, T = size(signal)

    # If spectrogram, drop DC-ish rows
    if N > 1
        signal = signal[2:end, :]

        signal = log.(signal)  # log transform
        signal[isnan.(signal)] .= 0  # remove nans
        
    	signal = signal .- sum(signal)/length(signal)
    	signal = max.(0, signal)
    	signal = signal / maximum(signal)
    end
    @assert !(any(isnan.(signal)))
    println("Data loaded.")

    # Fit data
    results = fit_cnmf(
        signal; 
        L=L, 
        K=K, 
        alg=ADMMUpdate, 
        l1H=beta, 
        max_itr=15, 
        max_time=Inf,
        verbose=true
    )

    # Save results
    println(round.(results.loss_hist; digits=4))
    matwrite(
        string(path, "trainout/", filename, "_", K, "_", L, ".mat"),
        Dict(
            "W" => results.W,
            "H" => results.H,
            "time_hist" => results.time_hist,
            "loss_hist" => results.loss_hist,
        )
    )
end



filename = ARGS[1]
K = 10
L = 30
beta = 0.5

runscript(filename, K, L, beta)
