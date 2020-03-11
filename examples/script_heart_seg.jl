using MAT
import PyPlot; plt = PyPlot


matdict = matread("/home/asd/data/heart/results/train_1.mat")
signal = matdict["signal"]
println(size(signal))

plt.figure()
plt.plot(signal[1, 1:500])
plt.show()