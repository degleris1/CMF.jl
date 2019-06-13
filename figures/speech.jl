
using Revise
using CMF
using PyPlot
using DSP
using WAV
using JLD
using ArgParse

path = "../../cmf_data/ira_glass.wav"
s, fs = WAV.wavread(path);

# Downsample to 8 Khz
fs_new = 8e3
p = Int(round(fs / fs_new))
s = s[1:p:end];


# Log transform spectrogram
start_idx = 400
end_idx = 800
S = spectrogram(s[:,1], 512, 384; window=hanning)
t = time(S)
f = freq(S)
data = log10.(S.power)
data = data .+ abs(minimum(data))

# Parse commandline arguments to allow specifying model size
s = ArgParseSettings()
@add_arg_table s begin
    "--alg"
        help = "Alg to use for speech fit"
        arg_type = String
        default = "mult"
end
parsed_args = parse_args(ARGS, s)

alg = Symbol(parsed_args["alg"])

# fit once to compile
CMF.fit_cnmf(data; L=12, K=20,
                       alg=alg, max_itr=1, max_time=6000,
                       check_convergence=false
                       )

results = CMF.fit_cnmf(data; L=12, K=20,
                       alg=alg, max_itr=1, max_time=6000,
                       check_convergence=false
                       )


alg_str = parsed_args["alg"]
JLD.save("speech_run_$alg_str.jld", "results", results)
