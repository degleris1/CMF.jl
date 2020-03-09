using MAT

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

    for (group, num) in keys(data)
        # Skip first individual per group
        if num == 1
            testusers[string(group, "_", num)] = data[(group, num)][dt]
        end
            
        # Load signal
        signal = data[(group, num)][dt]
        pad = zeros(size(signal, 1), padsize)

        # Downsample
        signal = signal[:, 1:downsample_rate:end]
        pad = pad[:, 1:downsample_rate:end]
        
        # Leave out a random 10% of the timebins
        # for test set (and store that information)
        T = size(signal, 2)
        tstart = rand(1:T)
        tend = min(T, tstart + floor(Int, T/10))
        
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
        string("/home/asd/data/heart/train_", dtname, ".mat"),
        Dict("signal" => train)
    )
    matwrite(
        string("/home/asd/data/heart/testusers_", dtname, ".mat"),
        Dict("testusers" => testusers)
    )
    matwrite(
        string("/home/asd/data/heart/testsegs_", dtname, ".mat"),
        Dict("testsegs" => testsegs)
    )

    println("Done.")
end

dt = ARGS[1]
dtname = ARGS[2]

runscript(dt, dtname)