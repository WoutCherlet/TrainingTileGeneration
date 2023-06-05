import os
import pandas as pd
from tqdm import tqdm
import open3d as o3d
import numpy as np


from utils import read_clouds, combine_pcds, get_bbox

DATA_DIR = "/home/wcherlet/data/Wytham2015"

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])



def write_bboxs(pc_folder, bbox_extent, bbox_trees):

    # Read all segmented trees
    pcds = read_clouds(pc_folder)

    # Compute overall bounding box
    get_bbox(pcds, compute_overall_bbox=True, path_out=bbox_extent)

    # Compute single bounding boxes
    get_bbox(pcds, path_out=bbox_trees)

    return

def clip_tiles(bbox_extent, valid_tiles_path, clipped_tiles_dir):
    bbox = pd.read_csv(bbox_extent)
    bb_x_min = bbox['x_min'].values[0]
    bb_x_max = bbox['x_max'].values[0]
    bb_y_min = bbox['y_min'].values[0]
    bb_y_max = bbox['y_max'].values[0]

    # Get tilenames
    with open(valid_tiles_path, 'r') as f:
        tilenames = f.readlines()

    # Define margin to clip tiles at border
    m = 10
    m_ymax = 25

    if not os.path.isdir(clipped_tiles_dir):
        os.mkdir(clipped_tiles_dir)

    # Loop over tiles and clip to bounding box + margin
    for i in tqdm(range(len(tilenames))):
        # Read tile
        tilename = tilenames[i]
        tilename = tilename.split('\n')[0]
        if not os.path.exists(os.path.join(DATA_DIR, "tiles", tilename)):
            print(f"can't find path {os.path.join(DATA_DIR, 'tiles', tilename)}")
            continue

        tile = o3d.io.read_point_cloud(os.path.join(DATA_DIR, "tiles", tilename), format="ply")
        
        # Convert points to numpy array 
        points = np.asarray(tile.points)
        x = points[:,0]
        y = points[:,1]

        # Select only points within bounding box + margin
        points = points[((x > (bb_x_min + m)) & (x < (bb_x_max - m)) & (y > (bb_y_min + m)) & (y < (bb_y_max - m_ymax))), :]

        # Save tile (if non empty)
        if points.shape[0] > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(os.path.join(clipped_tiles_dir, tilename), pcd)


def view1D(a, b): # a, b are arrays
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    return a.view(void_dt).ravel(),  b.view(void_dt).ravel()

def isin_nd(a,b):
    # a,b are the 3D input arrays to give us "isin-like" functionality across them
    A,B = view1D(a.reshape(a.shape[0],-1),b.reshape(b.shape[0],-1))
    return np.isin(A,B)

def isin_tolerance(A, B, tol):
    A = np.asarray(A)
    B = np.asarray(B)

    Bs = np.sort(B) # skip if already sorted
    idx = np.searchsorted(Bs, A)

    linvalid_mask = idx==len(B)
    idx[linvalid_mask] = len(B)-1
    lval = Bs[idx] - A
    lval[linvalid_mask] *=-1

    rinvalid_mask = idx==0
    idx1 = idx-1
    idx1[rinvalid_mask] = 0
    rval = A - Bs[idx1]
    rval[rinvalid_mask] *=-1
    return np.minimum(lval, rval) <= tol

def isclose_nd(a,b):
    A,B = view1D(a.reshape(a.shape[0],-1),b.reshape(b.shape[0],-1))
    return isin_tolerance(A,B, tol=0.01)

def get_understory(pc_folder, clipped_tiles_dir, bbox_trees):
    tilenames = [os.path.join(clipped_tiles_dir, f) for f in os.listdir(clipped_tiles_dir) if f[-3:] == 'ply']

    bbox = pd.read_csv(bbox_trees)

    # Make mapping of tree file to unique number 
    filenames = [f for f in os.listdir(pc_folder) if f[-3:] == 'ply']
    tree_file2number = {filename: i for i, filename in enumerate(filenames)}
    tree_number2file = {i: filename for i, filename in enumerate(filenames)}

    understory_tiles = []
    out_understory = os.path.join(DATA_DIR, "understory_tiles_close")
    if not os.path.exists(out_understory):
        os.mkdir(out_understory)

    # Iterate over all tiles
    for i in tqdm(range(len(tilenames))):

        # Read tile
        tilename = tilenames[i]
        tile = o3d.io.read_point_cloud(tilename)

        # Get bounds of tile
        points = np.asarray(tile.points)
        t_x_max = np.max(points[:,0])
        t_x_min = np.min(points[:,0])
        t_y_max = np.max(points[:,1])
        t_y_min = np.min(points[:,1])

        # Get trees that fall (partly) within bounds of the tile
        bbox_in = bbox[((bbox['x_min'] < t_x_max) & (bbox['x_max'] > t_x_min) & (bbox['y_min'] < t_y_max) & (bbox['y_max'] > t_y_min))]
        trees_names_in = [tree_number2file[i] for i in bbox_in.index]
        
        # Read in point clouds of included trees
        trees_in = read_clouds([os.path.join(pc_folder, tree_name) for tree_name in trees_names_in])

        # Pre-allocate label array with value '-1'
        tree_mask = np.zeros(len(points), dtype=np.int32)

        # Iterate over all trees within tile
        for tree_in in trees_in:
            # Boolean list indicating where tile points occur as tree points
            row_match = isclose_nd_slow(points, np.asarray(tree_in.points))

            # Allocate class and instance label
            tree_mask = np.logical_or(tree_mask, row_match)

        # get understory mask and slice points
        understory_mask = np.logical_not(tree_mask)
        understory_points = points[understory_mask]
        understory_cloud = o3d.geometry.PointCloud()
        understory_cloud.points = o3d.utility.Vector3dVector(understory_points)
        understory_tiles.append(understory_cloud)

        # write tiles also
        o3d.io.write_point_cloud(os.path.join(out_understory, f"understory_{os.path.basename(tilename)}"), understory_cloud)
        
        # TODO: temp: test on one tile
        return

    print("Merging understory tiles")

    combine_pcds(understory_tiles, path_out=os.path.join(DATA_DIR, "understory.ply"))

    return
            
def post_process_pc():
    understoryply = os.path.join(DATA_DIR, "understory.ply")

    understory_cloud = o3d.io.read_point_cloud(understoryply)

    # display inliers and outliers
    cl, ind = understory_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    display_inlier_outlier(understory_cloud, ind)

    # same with radius outlier detection 

    cl, ind = understory_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    display_inlier_outlier(understory_cloud, ind)



def main():
    # TODO: plan:
    # 2. Delete all tree points by checking segmeented trees
    # extra: remove terrain using some sort of filter

    pc_folder = os.path.join(DATA_DIR, "trees")
    bbox_extent = os.path.join(DATA_DIR, "extent.csv")
    bbox_trees = os.path.join(DATA_DIR, "bounding_boxes.csv")
    # Path to valid tiles within extent of segmented trees
    valid_tiles_path = os.path.join(DATA_DIR, 'valid_tiles.txt')
    clipped_tiles_dir = os.path.join(DATA_DIR, "tiles_clipped")

    # only run once!!
    # write_bboxs(pc_folder, bbox_extent, bbox_trees)

    # only run once!!
    # clip_tiles(bbox_extent, valid_tiles_path, clipped_tiles_dir)

    get_understory(pc_folder, clipped_tiles_dir, bbox_trees)

    return


if __name__ == "__main__":
    main()
