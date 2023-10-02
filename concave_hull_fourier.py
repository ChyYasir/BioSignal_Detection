import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
from scipy.spatial import ConvexHull
import alphashape
import math

class AlphaConcaveHull:
    def __init__(self, signal, alpha):
        self.signal = signal
        self.fourier_x = []
        self.fourier_y = []
        self.alpha = alpha

    def execute(self):
        fourier = fft(self.signal)

        self.fourier_x = [ele.real for ele in fourier]
        self.fourier_y = [ele.imag for ele in fourier]

        fourier_points = [[ele.real, ele.imag] for ele in fourier]
        arr_points = np.array(fourier_points)

        # Create the alpha shape
        alpha_shape = alphashape.alphashape(arr_points, self.alpha)

        # Extract the edges (LineString) of the alpha shape
        edges = alpha_shape.boundary

        # Extract the x and y coordinates of the edges
        x_edges, y_edges = edges.xy
        area = 0
        perimeter = 0
        n = len(x_edges)
        for i in range(0, n):
            area = area + abs((x_edges[i]* y_edges[(i+1)%n]) - (x_edges[(i+1)%n]*y_edges[i]))
            dis_x = (x_edges[(i+1)%n] - x_edges[i]) * (x_edges[(i+1)%n] - x_edges[i])
            dis_y = (y_edges[(i+1)%n] - y_edges[i]) * (y_edges[(i+1)%n] - y_edges[i])
            perimeter = perimeter + math.sqrt(dis_x+dis_y)
        area = area * 0.5
        pi = math.pi
        circularity = 4 * pi * area
        circularity = circularity / (perimeter * perimeter)
        print("Area = " + str(area))
        print("Perimeter = " + str(perimeter))
        print("Circularity = " + str(circularity))

        # Create a ConvexHull object from the points
        hull = ConvexHull(arr_points)

        # Extract the vertices of the convex hull
        hull_vertices = arr_points[hull.vertices]

        x_edges_convex = hull_vertices[:, 0]
        y_edges_convex = hull_vertices[:, 1]
        n = len(x_edges_convex)
        convex_perimeter = 0
        for i in range(0, n):

            dis_x = (x_edges_convex[(i+1)%n] - x_edges_convex[i]) * (x_edges_convex[(i+1)%n] - x_edges_convex[i])
            dis_y = (y_edges_convex[(i+1)%n] - y_edges_convex[i]) * (y_edges_convex[(i+1)%n] - y_edges_convex[i])
            convex_perimeter = convex_perimeter + math.sqrt(dis_x+dis_y)

        convexity = convex_perimeter / perimeter
        print("Convextiy = " + str(convexity))
        plt.scatter(self.fourier_x, self.fourier_y)
        plt.plot(x_edges, y_edges, 'k-', label='Alpha Shape Edges')
        plt.plot(hull_vertices[:, 0], hull_vertices[:, 1], 'r-', lw=2, label='Convex Hull')

