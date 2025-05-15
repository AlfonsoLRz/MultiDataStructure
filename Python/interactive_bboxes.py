import open3d as o3d
import numpy as np
import pandas as pd

folder = '../MultiDataStructure/output/'
file = 'nodes.csv'
df = pd.read_csv(folder + file)

def create_3d_box(bbox):
    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max]
    ])
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7]   # Sides
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set

boxes = [create_3d_box(bbox) for bbox in df[['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max']].values]
o3d.visualization.draw_geometries(boxes)