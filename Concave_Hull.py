import bisect
from collections import OrderedDict
import math
import matplotlib.tri as tri
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.ops import linemerge

class ConcaveHull:
    # Initialization
    def __init__(self):
        self.triangles = {}  # Dictionary to store triangles from triangulation
        self.crs = {}  # Not used in the given implementation

    # Load points into the class
    def loadpoints(self, points):
        self.points = points  # Store the points

    # Calculate the length of a triangle's edge not shared with any other triangles (boundary edge)
    def edge(self, key, triangle):
        '''Calculate the length of the triangle's outside edge
        and returns the [length, key]'''
        # Find the position of the -1 in the triangle neighbors, indicating an outside edge
        pos = triangle[1].index(-1)
        # Get the coordinates of the points forming the outside edge
        # and calculate the length of this edge
        if pos == 0:
            x1, y1 = self.points[triangle[0][0]]
            x2, y2 = self.points[triangle[0][1]]
        elif pos == 1:
            x1, y1 = self.points[triangle[0][1]]
            x2, y2 = self.points[triangle[0][2]]
        elif pos == 2:
            x1, y1 = self.points[triangle[0][0]]
            x2, y2 = self.points[triangle[0][2]]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return [length, key]

    # Perform Delaunay triangulation on the points
    def triangulate(self):
        if len(self.points) < 2:
            raise Exception('CountError: You need at least 3 points to Triangulate')

        # Extract x and y coordinates
        temp = list(zip(*self.points))
        x, y = list(temp[0]), list(temp[1])

        # Perform triangulation using matplotlib.tri
        triang = tri.Triangulation(x, y)

        # Store triangles and their neighbors
        for i, triangle in enumerate(triang.triangles):
            self.triangles[i] = [list(triangle), list(triang.neighbors[i])]

    # Calculate the concave hull by removing certain triangles
    def calculatehull(self, tol=50):
        self.tol = tol  # Tolerance for removing triangles

        if len(self.triangles) == 0:
            self.triangulate()

        deletion = []  # List to store triangles to be removed
        self.boundary_vertices = set()  # Set to store boundary vertices

        # Identify triangles with long boundary edges and add them to the deletion list
        for i, triangle in self.triangles.items():
            if -1 in triangle[1]:
                for pos, neigh in enumerate(triangle[1]):
                    if neigh == -1:
                        if pos == 0:
                            self.boundary_vertices.add(triangle[0][0])
                            self.boundary_vertices.add(triangle[0][1])
                        elif pos == 1:
                            self.boundary_vertices.add(triangle[0][1])
                            self.boundary_vertices.add(triangle[0][2])
                        elif pos == 2:
                            self.boundary_vertices.add(triangle[0][0])
                            self.boundary_vertices.add(triangle[0][2])
            if -1 in triangle[1] and triangle[1].count(-1) == 1:
                rec = self.edge(i, triangle)
                if rec[0] > self.tol and triangle[1].count(-1) == 1:
                    bisect.insort(deletion, rec)

        # Remove triangles from the deletion list
        while len(deletion) != 0:
            item = deletion.pop()  # Start with the longest edge
            ref = item[1]

            # Skip if triangle already has two boundary edges
            if self.triangles[ref][1].count(-1) > 1:
                continue

            # Check if the triangle should be deleted
            adjust = {0: 2, 1: 0, 2: 1}
            flag = 0
            for i, neigh in enumerate(self.triangles[ref][1]):
                j = adjust[i]
                if neigh == -1 and self.triangles[ref][0][j] in self.boundary_vertices:
                    flag = 1
                    break
            if flag == 1:
                continue

            # Update neighbors of the triangle being deleted
            for i, neigh in enumerate(self.triangles[ref][1]):
                if neigh == -1:
                    continue
                pos = self.triangles[neigh][1].index(ref)
                self.triangles[neigh][1][pos] = -1
                rec = self.edge(neigh, self.triangles[neigh])
                if rec[0] > self.tol and self.triangles[rec[1]][1].count(-1) == 1:
                    bisect.insort(deletion, rec)

            for pt in self.triangles[ref][0]:
                self.boundary_vertices.add(pt)

            del self.triangles[ref]

        # Create the concave hull polygon
        self.polygon()

    # Create a polygon from the remaining triangles
    def polygon(self):
        edgelines = []
        for i, triangle in self.triangles.items():
            if -1 in triangle[1]:
                for pos, value in enumerate(triangle[1]):
                    if value == -1:
                        if pos == 0:
                            x1, y1 = self.points[triangle[0][0]]
                            x2, y2 = self.points[triangle[0][1]]
                        elif pos == 1:
                            x1, y1 = self.points[triangle[0][1]]
                            x2, y2 = self.points[triangle[0][2]]
                        elif pos == 2:
                            x1, y1 = self.points[triangle[0][0]]
                            x2, y2 = self.points[triangle[0][2]]
                        line = LineString([(x1, y1), (x2, y2)])
                        edgelines.append(line)

        # Merge lines and create a polygon
        bound = linemerge(edgelines)
        self.boundary = Polygon(bound.coords)
