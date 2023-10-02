import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
from scipy.spatial import ConvexHull
import alphashape

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

        points = [[ele.real, ele.imag] for ele in fourier]
        arr_points = np.array(points)

        # Create the alpha shape
        alpha_shape = alphashape.alphashape(arr_points, self.alpha)

        # Extract the edges (LineString) of the alpha shape
        edges = alpha_shape.boundary

        # Extract the x and y coordinates of the edges
        x_edges, y_edges = edges.xy

        plt.scatter(self.fourier_x, self.fourier_y)
        plt.plot(x_edges, y_edges, 'k-', label='Alpha Shape Edges')


