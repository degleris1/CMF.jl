import PyPlot; plt = PyPlot
using FFTW

include("plot_settings.jl")

T = 100
L = 10

h = zeros(T)
h[20] = 1
h[40] = 2
h[70] = 1
h[75] = 1

w = zeros(T)
w[2:L] .= exp.(-20 * ( (1:L-1) .- L/2).^2 / L^2)

b = real.(ifft(fft(h) .* fft(w)))


# Generate plot
plt.figure(figsize=set_size())

plt.subplot(311)
plt.plot(1:T, w)
plt.title("\$w\$", pad=-20)
plt.xticks([])

plt.subplot(312)
markerline, stemline, baseline = plt.stem(1:T, h, basefmt=".")
plt.title("\$h\$", pad=-20)
plt.xticks([])
plt.setp(markerline, markersize=3)

plt.subplot(313)
plt.plot(1:T, b)
plt.title("\$b\$", pad=-20)

plt.savefig("./viz_conv.eps")