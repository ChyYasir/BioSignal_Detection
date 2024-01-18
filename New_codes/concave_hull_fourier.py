import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import alphashape
import math
import cv2
from Concave_Hull import ConcaveHull
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

    def features(self, x_edges, y_edges, rect):
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

        rect_width, rect_height = rect[1]
        rect_area = rect_width * rect_height
        # Rectangularity
        rectangularity = area / rect_area if rect_area != 0 else 0

        # Eccentricity
        longer_side = max(rect_width, rect_height)
        shorter_side = min(rect_width, rect_height)
        eccentricity = longer_side / shorter_side if shorter_side != 0 else 0
        return [area, perimeter, circularity, convexity, variance, bending_energy, rectangularity, eccentricity]
        # print("Variance = " , str(variance))
    def execute(self):
        fourier = fft(self.signal)
        fourier_points = [(point.real, point.imag) for point in fourier[1:]]
        self.fourier_x, self.fourier_y = zip(*fourier_points)
        arr_points = np.array(fourier_points)
        # print(fourier_points)

        ch = ConcaveHull()
        ch.loadpoints(fourier_points)
        ch.calculatehull(tol=10)
        ch.polygon()

        # boundary_points = np.vstack(ch.boundary.exterior.coords.xy).T
        boundary_points = np.array(ch.boundary.exterior.coords)
        boundary_x, boundary_y = boundary_points[:, 0], boundary_points[:, 1]
        # alpha = 0.95 * alphashape.optimizealpha(fourier_points)
        # concave_hull = alphashape.alphashape(arr_points, alpha)
        # concave_hull_pts = concave_hull.exterior.coords.xy


        hull_vertices = self.graham_scan(fourier_points)
        hull_vertices.append(hull_vertices[0])  # Closing the loop of the convex hull for plotting

        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot for Concave Hull
        axs[0].scatter(self.fourier_x, self.fourier_y)
        axs[0].plot(boundary_x, boundary_y, 'b-', label='Concave Hull')
        axs[0].set_title('Concave Hull')

        # Plot for Convex Hull
        x_edges_convex, y_edges_convex = zip(*hull_vertices)
        axs[1].scatter(self.fourier_x, self.fourier_y, label='Fourier Points')
        axs[1].plot(x_edges_convex, y_edges_convex, 'r-', label='Convex Hull')
        axs[1].set_title('Convex Hull')

        # Compute and draw the minimum bounding rectangle for convex hull
        rect = cv2.minAreaRect(np.array(hull_vertices, dtype=np.float32))
        box = cv2.boxPoints(rect)
        box = np.append(box, [box[0]], axis=0)
        axs[1].plot(box[:, 0], box[:, 1], 'g-', label='Bounding Rectangle')

        for ax in axs:
            ax.legend()
            ax.axis('equal')

        plt.show()

        features = self.features(x_edges_convex, y_edges_convex, rect)

        return features


