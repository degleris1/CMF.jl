using PyPlot: plt

using FFTW

include("plot_settings.jl")

T = 150
N = 5
K = 2
L = 10

h = zeros(K, T)
h[1, 25] = 1
h[1, 80] = 1
h[1, 144] = 1
h[2, 40] = 1
h[2, 100] = 1

w = zeros(K, N, T)
w[1, 1, 3:L+2] = (1:L) / L
w[1, 2, 3:L+2] = (L .- (1:L)) / L
w[1, 3, 3:L+2] = (1:L) / L
w[2, 3, 3:L+2] .= 1  #(L .- (1:L)) / L
w[2, 4, 3:Int(L/2)+2] .= 1 #  (1:L/2) / L
w[2, 5, Int(L/2)+1:L] .= 1 # (1:L/2) / L  
w /= 2.3

b = zeros(N, T)
for k = 1:K
    for n = 1:N
        b[n, :] += real.(ifft(fft(h[k, :]) .* fft(w[k, n, :])))
    end
end 


# Generate plot
plt.close()
fig = plt.figure(figsize=set_size())
nrows = 10
ncols = 10
offset = 1
shape = (nrows, ncols)
colors = ["r", "b"]

# Plot observed data
plt.subplot2grid(shape, (2, 2), rowspan=nrows-2, colspan=ncols-2)
plt.ylim(0.5, 6)
plt.yticks([])
plt.xticks([])
plt.xlabel("\$b_n = \\sum_k w_k * h_k \$")
for n = 1:N
    plt.plot(b[n, :] .+ offset*n, color="k")
end

# Plot motifs
for k = 1:2
    plt.subplot2grid(shape, (2, k-1), rowspan=nrows-2, colspan=1)
    plt.ylim(0.5, 6)
    plt.xticks([])
    k==1 && plt.yticks(1:5)
    k==1 && plt.ylabel("\$n\$")
    k==2 && plt.yticks([])
    plt.xlabel("\$w_{n" * string(k) * "}\$")
    for n = 1:N
        plt.plot(w[k, n, 1:L+10] .+ n, color=colors[k])
    end
end

# Plot feature maps
plt.subplot2grid(shape, (0, 2), rowspan=2, colspan=ncols-2)
plt.xticks([])
plt.yticks([2, 4.5], ["\$h_{2}\$", "\$h_{1}\$"])
plt.ylim(1, 6)
for k = 1:2
    plt.plot(1:T, h[k, :] .+ 6 .- 2k, color=colors[k])
end

plt.savefig("./viz_cf.eps")