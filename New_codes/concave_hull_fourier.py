import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import alphashape
import math
import cv2
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

    def orientation(self, p, q, r):
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    def graham_scan(self, points):

        min_y = min(points, key=lambda p: (p[1], p[0]))
        points.pop(points.index(min_y))
        # Sort the remaining point according to the polar angle with min_y
        points.sort(key=lambda p: (math.atan2(p[1] - min_y[1], p[0] - min_y[0]),
                                   (p[0] - min_y[0]) ** 2 + (p[1] - min_y[1]) ** 2))

        hull = [min_y, points[0], points[1]]
        for p in points[2:]:
            while len(hull) > 1 and self.orientation(hull[-2], hull[-1], p) >= 0:
                hull.pop()  # Remove last point from the hull if it's not making a left turn
            hull.append(p)
        return hull

    def features(self, x_edges, y_edges):
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

        n = len(x_edges)
        convex_perimeter = 0
        bending_energy = 0
        omega = (2 * pi) / perimeter
        for i in range(0, n):
            dis_x = (x_edges[(i+1)%n] - x_edges[i]) * (x_edges[(i+1)%n] - x_edges[i])
            dis_y = (y_edges[(i+1)%n] - y_edges[i]) * (y_edges[(i+1)%n] - y_edges[i])
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
        return [area, perimeter, circularity, convexity, variance, bending_energy]
        # print("Variance = " , str(variance))
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

        plt.scatter(self.fourier_x, self.fourier_y)
        # plt.show()
        # print(arr_points)
        # Create the alpha shape
        alpha_shape = alphashape.alphashape(arr_points, self.alpha)
        #
        # Extract the edges (LineString) of the alpha shape
        edges = alpha_shape.boundary
        #
        # # # Extract the x and y coordinates of the edges
        # x_edges, y_edges = edges.xy
        #
        # Create a ConvexHull object from the points
        hull_vertices = self.graham_scan(fourier_points)

        # Closing the loop of the convex hull for plotting
        hull_vertices.append(hull_vertices[0])

        # Plot the convex hull
        x_edges_convex, y_edges_convex = [], []
        for vertex in hull_vertices:
            x_edges_convex.append(vertex[0])
            y_edges_convex.append(vertex[1])

        plt.plot(x_edges_convex, y_edges_convex, 'r-', label='Convex Hull')

        # Compute the minimum bounding rectangle
        rect = cv2.minAreaRect(np.array(hull_vertices, dtype=np.float32))

        print(rect)
        # Get the rectangle's parameters
        rect_center, rect_size, rect_angle = rect

        # Draw the minimum bounding rectangle
        box = cv2.boxPoints(rect)
        box = np.append(box, [box[0]], axis=0)

        # box = np.int0(box)
        plt.plot(box[:, 0], box[:, 1], 'g-', label='Bounding Rectangle')

        # ... (rest of your code)

        plt.legend()
        plt.show()
        features = self.features(x_edges_convex, y_edges_convex)

        return features


