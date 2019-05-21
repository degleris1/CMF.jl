function generate_logspect(path, freq; seconds=nothing)
    # Load raw file
    raw, f = WAV.wavread(path)

    # Merge channels
    stream = raw[:, 1] + raw[:, 2]

    # Downsmaple
    down_rate = Integer(round(f / freq))
    downstream = stream[1 : down_rate : end]

    # Crop
    if (seconds == nothing)
        cropped = downstream
    else
        last = seconds * freq
        cropped = downstream[1:min(last, length(downstream))]
    end 

    # Create spectrogram
    spect = DSP.spectrogram(cropped,
                            1024, 512, window=DSP.hamming).power

    # Log transform
    logspect = log.(spect) .- minimum(log.(spect))

    return logspect
end


"""
Piano dataset.
First 30 seconds of Bach Prelude and Fugue No. 1 in D Major
Stored in a wave file sampled at 44100 Hz

Reference
---------
https://www.youtube.com/watch?v=3srDTD2M8ol
"""
function piano(;path="/home/anthony/cmf_data/prelude_bach.wav",
               freq=11025, seconds=30)
    return generate_logspect(path, freq, seconds=seconds)
end



"""
Female voices
"""
function female_voices(;path="/home/anthony/cmf_data/test2_female3_inst_mix.wav",
                       freq=11025)
    return generate_logspect(path, freq)
end


"""
Drums mix
"""
function drums_mix(;path="/home/anthony/cmf_data/test2_wdrums_inst_mix.wav",
                   freq=11025)
    return generate_logspect(path, freq)
end


