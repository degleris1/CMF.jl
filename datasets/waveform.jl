import CMF
import PyPlot
const plt = PyPlot

function triangle(n)
    x = zeros(2*n)
    scale = 1 / n
    x[1:n] = scale * (1:n)
    x[n+1:2*n] = 1 .- scale * (1:n)
    return x
end

function generate_trianglewave()
    x = zeros(50)

    x[16:35] = triangle(10)

    return [x; x; x]
end

function disp_trianglewave()
    x = generate_trianglewave()
    plt.plot(x, markersize=4, marker="o")
end

function generate_heartbeat()
    x = zeros(60)

    x[11:14] = 0.5 * triangle(2)
    x[15:22] = -3 * triangle(4)
    x[23:32] = 4 * triangle(5)
    x[33:38] = -triangle(3)
    x[39:44] = triangle(3)

    return [x; x; x]
end

function disp_heartbeat()
    x = generate_heartbeat()
    plt.plot(x, markersize=2, marker="o")
end

disp_heartbeat()