using MAT
using CMF

# Desired settings
# K = 10, L = 30
# K = 20, L = 15
# t32 10 30 --- 12:41

function runscript(filename, K, L, beta)
    folder = "/farmshare/user_data/degleris/heart/"
    matdata = matread(string(folder, filename, ".mat"))
    signal = matdata["signal"]

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
    println("Data loaded.")

    # Fit data
    results = fit_cnmf(
        signal; 
        L=L, 
        K=K, 
        alg=ADMMUpdate, 
        l1H=beta, 
        max_itr=20, 
        max_time=Inf,
        verbose=true
    )

    # Save results
    println(round.(results.loss_hist; digits=4))
    matwrite(
        string(folder, filename, "_K", K, "_L", L, "_B", beta, "_results.mat"),
        Dict(
            "W" => results.W,
            "H" => results.H,
            "time_hist" => results.time_hist,
            "loss_hist" => results.loss_hist,
        )
    )
end


@assert length(ARGS) >= 3

filename = ARGS[1]
K = parse(Int64, ARGS[2])
L = parse(Int64, ARGS[3])
if length(ARGS) >= 4
    beta = parse(Float64, ARGS[4])
else
    beta = 0.25
end

runscript(filename, K, L, beta)
