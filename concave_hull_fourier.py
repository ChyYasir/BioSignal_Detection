import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import alphashape
import math

class AlphaConcaveHull:
    def __init__(self, signal, alpha):
        self.signal = signal
        self.fourier_x = []
        self.fourier_y = []
        self.alpha = alpha

    def dist_between_two_point(self, x1, y1, x2, y2):
        dis_x = (x1 - x2) * (x1-x2)
        dis_y = (y1- y2) * (y1 - y2)
        ans = math.sqrt(dis_x + dis_y)
        return ans
    def execute(self):
        fourier = fft(self.signal)

        # self.fourier_x = [ele.real for ele in fourier]
        # self.fourier_y = [ele.imag for ele in fourier]
        #
        # fourier_points = [[ele.real, ele.imag] for ele in fourier]
        fourier_points = []
        for i in range(len(fourier)):
            if i == 0:
                continue
            self.fourier_x.append(fourier[i].real)
            self.fourier_y.append(fourier[i].imag)
            fourier_points.append([fourier[i].real, fourier[i].imag])
        arr_points = np.array(fourier_points)
        # print(arr_points)
        # Create the alpha shape
        alpha_shape = alphashape.alphashape(arr_points, self.alpha)

        # Extract the edges (LineString) of the alpha shape
        edges = alpha_shape.boundary

        # # Extract the x and y coordinates of the edges
        x_edges, y_edges = edges.xy




        area = 0
        perimeter = 0
        n = len(x_edges)

        vertices = []
        for i in range(0, n):
            vertices.append((x_edges[i], y_edges[i]))

            area = area + abs((x_edges[i]* y_edges[(i+1)%n]) - (x_edges[(i+1)%n]*y_edges[i]))
            dis_x = (x_edges[(i+1)%n] - x_edges[i]) * (x_edges[(i+1)%n] - x_edges[i])
            dis_y = (y_edges[(i+1)%n] - y_edges[i]) * (y_edges[(i+1)%n] - y_edges[i])
            perimeter = perimeter + math.sqrt(dis_x+dis_y)
        area = area * 0.5
        pi = math.pi
        circularity = 4 * pi * area
        circularity = circularity / (perimeter * perimeter)
        # print("Area = " + str(area))
        # print("Perimeter = " + str(perimeter))
        # print("Circularity = " + str(circularity))

        # Create a ConvexHull object from the points
        hull = ConvexHull(arr_points)

        # Extract the vertices of the convex hull
        hull_vertices = arr_points[hull.vertices]

        x_edges_convex = hull_vertices[:, 0]
        y_edges_convex = hull_vertices[:, 1]
        n = len(x_edges_convex)
        convex_perimeter = 0
        bending_energy = 0
        omega = (2 * pi) / perimeter
        for i in range(0, n):
            dis_x = (x_edges_convex[(i+1)%n] - x_edges_convex[i]) * (x_edges_convex[(i+1)%n] - x_edges_convex[i])
            dis_y = (y_edges_convex[(i+1)%n] - y_edges_convex[i]) * (y_edges_convex[(i+1)%n] - y_edges_convex[i])
            convex_perimeter = convex_perimeter + math.sqrt(dis_x + dis_y)

        n = len(x_edges)
        for i in range(0, n):
            bending_energy = bending_energy + ((x_edges[i]**2) + (y_edges[i]**2))

        bending_energy = bending_energy * ((perimeter**2)/n)
        convexity = convex_perimeter / perimeter
        # print("Convextiy = " + str(convexity))
        # print("Bending Energy = "+ str(bending_energy))

        # For variance calculation we have to determine the centroid

        # print(vertices)
        polygon = Polygon(vertices)
        # Get the centroid
        centroid = polygon.centroid

        # Access the (x, y) coordinates of the centroid
        centroid_x, centroid_y = centroid.x, centroid.y

        # print("Centroid coordinates (x, y):", centroid_x, centroid_y)
        n = len(x_edges)
        variance = 0
        for i in range(n):
            centroid_to_current = self.dist_between_two_point(centroid_x, centroid_y, x_edges[i], y_edges[i])
            variance = variance + (centroid_to_current * centroid_to_current)

        variance = variance / n

        # print("Variance = " , str(variance))

        # plt.scatter(self.fourier_x, self.fourier_y)
        # plt.scatter(x_edges, y_edges)
        # plt.plot(x_edges, y_edges, 'k-', label='Alpha Shape Edges')
        # # plt.plot(hull_vertices[:, 0], hull_vertices[:, 1], 'r-', lw=2, label='Convex Hull')
        #
        # plt.show()
        features = [area, perimeter, circularity, convexity, variance, bending_energy]
        return features


