using MAT
using CMF

function runscript(motif_filename, test_filename, out_filename)
    # Output dictionary
    outdict = Dict()

    # Load in motifs
    motifdict = matread(motif_filename)
    W = motifdict["W"]

    # Load amd fit test data
    testdict_ext = matread(test_filename)
    testdict = testdict_ext["testusers"]
    for k in keys(testdict)
        println("Fitting ", k)
        data = testdict[k]
        
        # Preprocess (drop DC and downsample)
        # if size(data, 1) > 1
        #     data = data[2:end, :]
        # end
        # data = data[:, 1:10:end]

        outdict[k] = evaluate_feature_maps(data, W)
    end
        
    matwrite(out_filename, outdict)
end

folder = "/home/asd/data/heart/results/"

motif_filename = string(folder, ARGS[1]) 
test_filename = string(folder, ARGS[2])
if length(ARGS) >= 3
    out_filename = ARGS[3]
else
    out_filename = string(motif_filename[1:end-4], "_usereval.mat")
end

runscript(motif_filename, test_filename, out_filename)