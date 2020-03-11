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

    path = "/home/asd/data/heart/"


    data = Dict()

    for (name, num) in folders
        println("Loading ", name)
        for i = 1:num
            data[(name, i)] = matread(string(
                    path, "raw/", name, "/", lowercase(name), "_", i, ".mat"
            ))
        end
    end

    groups = unique([k[1] for k in keys(data)])
    matwrite(path * "groups.mat", Dict("groups" => groups))

    downsample_rate = 10
    padsize = 1000

    testusers = Dict()

    for gr in groups
        # Initialize stuff

        train = []
        trainbreaks = []
        userorder = []
        
        testsegs = Dict()
        testbreaks = Dict()

        for (group, num) in keys(data)
            if group != gr
                continue
            end

            # Load signal
            signal = data[(group, num)][dt]
            pad = zeros(size(signal, 1), padsize)

            # Downsample
            signal = signal[:, 1:downsample_rate:end]
            pad = pad[:, 1:downsample_rate:end]

            # Skip first individual per group
            if num == 1
                testusers[group] = signal
                continue
            end

            push!(trainbreaks, size(train, 2)+1)
            push!(userorder, num)
            
            # Leave out a random 10% of the timebins
            # for test set (and store that information)
            T = size(signal, 2)
            tstart = rand(1:T)
            tend = min(T, tstart + floor(Int, T/10))

            # Store test segment
            testbreaks[string("n", num)]= [1 tstart tend T]
            testsegs[string("n", num)] = signal[:, tstart:tend]

            # Store train data
            signal = [signal[:, 1:tstart] signal[:, tend:end]]     
            if length(train) == 0
                train = signal
            else
                train = [train pad signal]
            end
        end

        # Save files
        matwrite(
            path * "train/" * string(dtname) * gr * ".mat",
            Dict("data" => train, "breaks" => trainbreaks, "labels" => userorder)
        )
        matwrite(
            path * "test/" * string(dtname) * gr * ".mat",
            Dict("segs" => testsegs, "breaks" => testbreaks, "user" => testusers[gr])
        )
    end
    

    println("Done.")
end

dt = ARGS[1]
dtname = ARGS[2]

runscript(dt, dtname)