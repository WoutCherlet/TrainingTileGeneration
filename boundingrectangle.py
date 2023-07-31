import os

import numpy as np
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt

def boundingrectangle(points, buffer: int = None, visualisation: bool = False, out_dir = None):
    """
    Calculates smallest bounding rectangle aroung set of 2D points

    Parameters
    ----------
    points
        array of 2D points
    (optional) buffer
        optional buffer to add to edge of bounding rectangle
    (optional) visualisation
        wether to show figure with bounding box
    (optional) out_dir
        location to save figure if provided
    """
    
    points = np.asarray(points)
    hull = ConvexHull(points) # scipy fast convex hull

    if buffer is None:
        buffer = 0

    min_area = np.inf
    final_corners = None

    # TODO: could do this all in matrix multiplications instead of with for loop
    for simplex in hull.simplices:
        start = points[simplex[0]]
        end = points[simplex[1]]

        # get unit direction and normal of current simplex
        direction = np.sum((end, -start), axis=0)
        norm = np.sqrt(np.sum(direction**2))
        unit_dir = direction/norm
        normal_dir = np.asarray([-unit_dir[1], unit_dir[0]])

        # tranform vertices to align current simplex with x axis
        vertices = points[hull.vertices]
        trans_matr = np.vstack((unit_dir, normal_dir)).T
        transformed_vertices = np.matmul(vertices, trans_matr)

        # get minimal and maximal x and y values
        min_vals = np.amin(transformed_vertices, axis=0)
        max_vals = np.amax(transformed_vertices, axis=0)

        max_vals += buffer/2
        min_vals -= buffer/2
        # calculate area
        area = np.prod(max_vals-min_vals)
        # get corner points
        corner_points = np.array([[min_vals[0], min_vals[1]], [min_vals[0], max_vals[1]], [max_vals[0], max_vals[1]], [max_vals[0], min_vals[1]]])

        if area < min_area:
            min_area = area
            # retransform corner points to get edge points using inverse of previous matrix multiplication
            # unit vectors so no renorm, and can just use T
            inv_matrix = trans_matr.T
            final_corners = np.matmul(corner_points, inv_matrix)
    
    if (visualisation or out_dir):
        plt.figure()
        plt.axis('equal')

        for point in points:
            plt.plot(point[0], point[1], 'bo')
        for point in final_corners:
            plt.plot(point[0], point[1], 'ro')
        # create edges:
        for i in range(len(final_corners)-1):
            plt.plot([final_corners[i][0], final_corners[i+1][0]], [final_corners[i][1], final_corners[i+1][1]], 'r-')
        # final line
        plt.plot([final_corners[-1][0], final_corners[0][0]], [final_corners[-1][1], final_corners[0][1]], 'r-')
        if(out_dir):
            if not os.path.exists(out_dir):
                print("Cant find location {out_dir} to save Plotbounds")
            else:
                plt.savefig(os.path.join(out_dir,"PlotBounds.png"))
        if (visualisation):
            plt.show()

    return min_area, final_corners
