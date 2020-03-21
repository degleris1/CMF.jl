using MAT
import PyPlot; plt = PyPlot

function runscript()
    path = "/home/asd/data/heart/"
    md_groups = matread(path * "groups.mat")
    groups = md_groups["groups"]
end