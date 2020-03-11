using MAT
using Random

# julia script_heart_preprocess.jl dat 1
# julia script_heart_preprocess.jl spec_w_64 32
# julia script_heart_preprocess.jl spec_w_128 128
# julia script_heart_preprocess.jl spec_w_256 256


Random.seed!(1234)

function runscript(dt, dtname)
    folders = [
        ("AFIB", 6), ("AFL", 3), ("APB", 9), ("Bigeminy", 7),
        ("LBBB", 3), ("NSR", 23), ("RBBB", 3), ("Trigeminy", 4)
    ];

    data = Dict()

    for (name, num) in folders
        println("Loading ", name)
        for i = 1:num
            data[(name, i)] = matread(string(
                    "/home/asd/data/heart/", name, "/", 
                    lowercase(name), "_", i, ".mat"
            ))
        end
    end


    downsample_rate = 10
    padsize = 1000

    train = []
    testusers = Dict()
    testsegs = Dict()

    train_breaks = []
    test_breaks = []
    user_order = []

    for (group, num) in keys(data)
        # Load signal
        signal = data[(group, num)][dt]
        pad = zeros(size(signal, 1), padsize)

        # Downsample
        signal = signal[:, 1:downsample_rate:end]
        pad = pad[:, 1:downsample_rate:end]

        # Drop DC and normalize
        if size(signal, 1) > 1
            print(".")
            signal = signal[2:end, :]
            pad = pad[2:end, :]
            signal = signal / maximum(signal)
        end

        # Skip first individual per group
        if num == 1
            testusers[string(group, "_", num)] = signal
            continue
        end

        push!(train_breaks, size(train, 2)+1)
        push!(user_order, string(group, "_", num))
        
        # Leave out a random 10% of the timebins
        # for test set (and store that information)
        T = size(signal, 2)
        tstart = rand(1:T)
        tend = min(T, tstart + floor(Int, T/10))
        push!(test_breaks, string(1, "_", tstart, "_", tend, "_", T))
        
        testsegs[string(group, "_", num)] = signal[:, tstart:tend]
        signal = [signal[:, 1:tstart] signal[:, tend:end]]     
        
        if length(train) == 0
            train = signal
        else
            train = [train pad signal]
        end
    end

    # Save files
    matwrite(
        string("/home/asd/data/heart/results/train_", dtname, ".mat"),
        Dict("signal" => train)
    )
    matwrite(
        string("/home/asd/data/heart/results/testusers_", dtname, ".mat"),
        Dict("testusers" => testusers)
    )
    matwrite(
        string("/home/asd/data/heart/results/testsegs_", dtname, ".mat"),
        Dict("testsegs" => testsegs)
    )
    matwrite(
        string("/home/asd/data/heart/results/breaks_", dtname, ".mat"),
        Dict("train_breaks" => train_breaks, "test_breaks" => test_breaks)
    )
    matwrite(
        string("/home/asd/data/heart/results/order.mat"),
        Dict("trainorder" => user_order)
    )

    println("Done.")
end

dt = ARGS[1]
dtname = ARGS[2]

runscript(dt, dtname)