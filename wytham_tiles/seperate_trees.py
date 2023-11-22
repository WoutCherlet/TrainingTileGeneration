import os
import pandas as pd
from tqdm import tqdm
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from utils import read_clouds, combine_pcds, get_bbox

DATA_DIR = "/home/wcherlet/data/singletrees/Wytham2015"


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
    # TODO: problem with A and B being 2d arrays, need some way to do this using xyz elements
    A = np.asarray(A)
    B = np.asarray(B)

    Bs = np.sort(B) # skip if already sorted
    idx = np.searchsorted(Bs, A)
    # searchsorted returns for each element in A the index where element should be inserted to maintain order
    # aka idx[i] satisfies Bs[idx[i] - 1] < A[i] < Bs[idx[i]]

    linvalid_mask = idx==len(B) # if idx is len(B), the element of A is bigger then all elements in B, but we still want to check if we are close to the last element of B
    idx[linvalid_mask] = len(B)-1 # if we are at the end of the list, check the last element of B
    lval = Bs[idx] - A # get the difference between the closest element of B on the right (bigger) and corresponding element of A
    lval[linvalid_mask] *=-1 # where we were at the end of the array, A was larger then B so switch sign

    rinvalid_mask = idx==0 # if idx is 0, the element of A is smaller then all elements of B, but we still want to check if we are close to the first element of B
    idx1 = idx-1 # substract 1 of index, to get the left closest element
    idx1[rinvalid_mask] = 0 # if idx is 0, add one back to get leftmost element of B
    rval = A - Bs[idx1] # get the difference between closest element of B on the left (smaller) and corresponding element of A
    rval[rinvalid_mask] *=-1 # where we were at start of array, A was smaller then B so switch sign
    return np.minimum(lval, rval) <= tol # return a boolean array of A.shape where A is within tol distance of any element of B

def isclose_nd(a,b):
    a_rounded = np.around(a, decimals=2)
    b_rounded = np.around(b, decimals=2)
    return isin_nd(a_rounded,b_rounded)

def get_understory(pc_folder, clipped_tiles_dir, bbox_trees):
    tilenames = [os.path.join(clipped_tiles_dir, f) for f in os.listdir(clipped_tiles_dir) if f[-3:] == 'ply']

    bbox = pd.read_csv(bbox_trees)

    # Make mapping of tree file to unique number 
    filenames = [f for f in os.listdir(pc_folder) if f[-3:] == 'ply']
    tree_file2number = {filename: i for i, filename in enumerate(filenames)}
    tree_number2file = {i: filename for i, filename in enumerate(filenames)}

    understory_tiles = []

    out_seperated = os.path.join(DATA_DIR, "seperated")
    if not os.path.exists(out_seperated):
        os.mkdir(out_seperated)

    # directories for all seperated tiles, close and exact matches
    out_understory = os.path.join(out_seperated, "understory_tiles_close")
    if not os.path.exists(out_understory):
        os.mkdir(out_understory)
    out_understory_exact = os.path.join(out_seperated, "understory_tiles")
    if not os.path.exists(out_understory_exact):
        os.mkdir(out_understory_exact)
    out_trees = os.path.join(out_seperated, "trees_tiles_close")
    if not os.path.exists(out_trees):
        os.mkdir(out_trees)
    out_trees_exact = os.path.join(out_seperated, "trees_tiles")
    if not os.path.exists(out_trees_exact):
        os.mkdir(out_trees_exact)

    cur_instance = 1

    # Iterate over all tiles
    for i in range(len(tilenames)):

        # Read tile
        tilename = tilenames[i]
        tile = o3d.t.io.read_point_cloud(tilename)
        print(f"Seperating {tilename} ({i+1}/{len(tilenames)})")

        # Get bounds of tile
        points = tile.point.positions.numpy()
        t_x_max = np.max(points[:,0])
        t_x_min = np.min(points[:,0])
        t_y_max = np.max(points[:,1])
        t_y_min = np.min(points[:,1])

        # Get trees that fall (partly) within bounds of the tile
        bbox_in = bbox[((bbox['x_min'] < t_x_max) & (bbox['x_max'] > t_x_min) & (bbox['y_min'] < t_y_max) & (bbox['y_max'] > t_y_min))]
        trees_names_in = [tree_number2file[i] for i in bbox_in.index]

        # Read in point clouds of included trees
        trees_in = read_clouds([os.path.join(pc_folder, tree_name) for tree_name in trees_names_in])

        total_tree_mask = np.zeros(len(points), dtype=np.int32)
        total_tree_mask_exact = np.zeros(len(points), dtype=np.int32)

        instance_masks = {}

        # Iterate over all trees within tile
        for tree_name, tree_in in zip(trees_names_in, trees_in):
            # Boolean list indicating where tile points occur as tree points
            row_match = isclose_nd(points, np.asarray(tree_in.points))

            row_match_exact = isin_nd(points, np.asarray(tree_in.points))

            # get total tree mask for total
            total_tree_mask = np.logical_or(total_tree_mask, row_match)
            total_tree_mask_exact = np.logical_or(total_tree_mask_exact, row_match_exact)

            instance_masks[tree_file2number[tree_name]] = row_match
            cur_instance += 1

        # get understory mask and slice points
        understory_mask = np.logical_not(total_tree_mask)
        understory_points = points[understory_mask]
        understory_cloud = o3d.t.geometry.PointCloud()
        understory_cloud.point.positions = o3d.core.Tensor(understory_points)
        understory_tiles.append(understory_cloud)
        o3d.t.io.write_point_cloud(os.path.join(out_understory, f"understory_{os.path.basename(tilename)}"), understory_cloud)

        understory_mask_exact = np.logical_not(total_tree_mask_exact)
        understory_points_exact = points[understory_mask_exact]
        understory_cloud_exact = o3d.t.geometry.PointCloud()
        understory_cloud_exact.point.positions = o3d.core.Tensor(understory_points_exact)
        o3d.t.io.write_point_cloud(os.path.join(out_understory, f"understory_{os.path.basename(tilename)}"), understory_cloud)

        # also label tree point clouds with labels
        instance_labels = np.zeros(len(points), dtype=np.int32)
        for instance in instance_masks:
            mask = instance_masks[instance]
            instance_labels[mask] = instance

        tree_points = points[total_tree_mask]
        usable_labels = instance_labels[total_tree_mask]
        tree_cloud = o3d.t.geometry.PointCloud()
        tree_cloud.point.positions = o3d.core.Tensor(tree_points)
        tree_cloud.point.instance_labels = o3d.core.Tensor(usable_labels[:, np.newaxis]) # append newaxis to get shape (n_points, 1)
        o3d.t.io.write_point_cloud(os.path.join(out_trees, f"trees_{os.path.basename(tilename)}"), tree_cloud)

        tree_points_exact = points[total_tree_mask_exact]
        tree_cloud_exact = o3d.t.geometry.PointCloud()
        tree_cloud_exact.point.positions = o3d.core.Tensor(tree_points_exact)
        o3d.t.io.write_point_cloud(os.path.join(out_trees_exact, f"trees_{os.path.basename(tilename)}"), tree_cloud_exact)

    # print("Merging understory tiles")

    # combine_pcds(understory_tiles, path_out=os.path.join(DATA_DIR, "understory.ply"))

    return


def kd_tree_redivision_single(tilename, understory_cloud, tree_cloud):

    understory_points = understory_cloud.point.positions.numpy()
    tree_points = tree_cloud.point.positions.numpy()

    understory_instance_labels = -1*np.ones(understory_points.shape[:1], dtype=np.int32)
    tree_instance_labels = tree_cloud.point.instance_labels.numpy()

    understory_labels = np.zeros(understory_points.shape[:1], dtype=np.int32)
    tree_labels = np.ones(tree_points.shape[:1], dtype=np.int32)

    all_points = np.concatenate([understory_points, tree_points])
    all_labels = np.concatenate([understory_labels, tree_labels])

    corrected_labels = np.copy(all_labels)

    all_instance_labels = np.concatenate([understory_instance_labels, tree_instance_labels.flatten()])
    corrected_instance_labels = np.copy(all_instance_labels)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(all_points)

    print("Creating KDTree")
    pcd_tree = o3d.geometry.KDTreeFlann(pc)

    print("Looping over understory points")
    N_CLOSEST = 500
    CUTOFF = N_CLOSEST / 2
    switching_n = 0
    for i, point in enumerate(understory_points):

        [_, idx, _] = pcd_tree.search_knn_vector_3d(point, N_CLOSEST)

        closest_labels = all_labels[idx]
        if np.sum(closest_labels) > CUTOFF:
            corrected_labels[i] = 1
            switching_n += 1
            # if we switch, also get closest instance labels
            closest_instance_labels = all_instance_labels[idx]
            closest_tree_instances = closest_instance_labels[closest_instance_labels != -1] # -1 can still be most frequent
            values, counts = np.unique(closest_tree_instances, return_counts=True)
            if len(counts) == 0:
                print("DEBUG: got 0-length array for counts, shouldn't happen, skipping this point")
                print(closest_instance_labels)
                print(closest_tree_instances)
                continue
            most_frequent = values[np.argmax(counts)]
            corrected_instance_labels[i] = most_frequent


        if (i+1) % 100000 == 0:
            print(f"({i+1}/{len(understory_points)})")

        
    print(f"Switched {switching_n} of {len(understory_points)} understory points.")

    understory_mask = np.logical_not(corrected_labels)
    understory_kd_points = all_points[understory_mask]
    understory_kd = o3d.geometry.PointCloud()
    understory_kd.points = o3d.utility.Vector3dVector(understory_kd_points)

    tree_kd_points = all_points[corrected_labels.astype(bool)]
    tree_instance_labels = corrected_instance_labels[corrected_labels.astype(bool)]
    tree_kd = o3d.t.geometry.PointCloud()
    tree_kd.point.positions = o3d.core.Tensor(tree_kd_points)
    tree_kd.point.instance_labels = o3d.core.Tensor(tree_instance_labels[:,np.newaxis])

    tiles_kd_understory = os.path.join(DATA_DIR, "seperated", "understory_kd")
    if not os.path.exists(tiles_kd_understory):
        os.mkdir(tiles_kd_understory)
    tiles_kd_trees = os.path.join(DATA_DIR, "seperated", "trees_kd")
    if not os.path.exists(tiles_kd_trees):
        os.mkdir(tiles_kd_trees)

    o3d.io.write_point_cloud(os.path.join(tiles_kd_understory, tilename + "_understory_kd.ply"), understory_kd)
    o3d.t.io.write_point_cloud(os.path.join(tiles_kd_trees, tilename + "_tree_kd.ply"), tree_kd)
    
    return


def kd_tree_redivision_all(clipped_tiles_dir, skip=True):
    tilenames = [f for f in sorted(os.listdir(clipped_tiles_dir)) if f[-3:] == 'ply']

    understory_dir = os.path.join(DATA_DIR, "seperated", "understory_tiles_close")
    trees_dir = os.path.join(DATA_DIR, "seperated", "trees_tiles_close")

    for i, tilename in enumerate(tilenames):
        print(f"Performing kd tree redivision for tile {tilename} ({i+1}/{len(tilenames)})")

        if skip and os.path.exists(os.path.join(DATA_DIR, "seperated", "understory_kd", tilename + "_understory_kd.ply")):
            print(f"Skipping tile {tilename} as kd files already exist")
            continue

        if not os.path.exists(os.path.join(understory_dir, "understory_" + tilename)) or not os.path.exists(os.path.join(trees_dir, "trees_" + tilename)):
            print(f"Trees and/or understory file not found for tilename {tilename}")
            continue
        
        understory_cloud = o3d.t.io.read_point_cloud(os.path.join(understory_dir, "understory_" + tilename))
        understory_cloud = understory_cloud.voxel_down_sample(voxel_size=0.03)
        tree_cloud = o3d.t.io.read_point_cloud(os.path.join(trees_dir, "trees_" + tilename))
        tree_cloud = tree_cloud.voxel_down_sample(voxel_size=0.03)
        
        kd_tree_redivision_single(os.path.basename(tilename), understory_cloud, tree_cloud)


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def post_process_pc():
    understoryply = os.path.join(DATA_DIR, "understory.ply")
    # TODO: temp
    understoryply = os.path.join(DATA_DIR, "understory_tiles_close", "understory_wytham_winter_102.ply")

    understory_cloud = o3d.io.read_point_cloud(understoryply)

    # display inliers and outliers
    cl, ind = understory_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    display_inlier_outlier(understory_cloud, ind)


def get_xy_view(clipped_tiles_dir):
    tilenames = [f for f in sorted(os.listdir(clipped_tiles_dir)) if f[-3:] == 'ply']
    
    x = []
    y = []
    for tilename in tilenames:
        file = os.path.join(clipped_tiles_dir, tilename)

        pc = o3d.io.read_point_cloud(file)

        center = (pc.get_max_bound() + pc.get_min_bound())/2

        x.append(center[0])
        y.append(center[1])

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(tilenames):
        ax.annotate(txt, (x[i], y[i]))

    plt.show()
    


def main():
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

    # get_understory(pc_folder, clipped_tiles_dir, bbox_trees)

    # kd_tree_redivision_all(clipped_tiles_dir)

    # get_xy_view(clipped_tiles_dir)

    # single tile: DEBUG
    # tile_name = "wytham_winter_122.ply"

    # out_understory = os.path.join(DATA_DIR, "seperated", "understory_tiles_close")
    # understory_cloud = o3d.t.io.read_point_cloud(os.path.join(out_understory, "understory_" + tile_name))
    # understory_cloud = understory_cloud.voxel_down_sample(voxel_size=0.03)
    # out_trees = os.path.join(DATA_DIR, "seperated", "trees_tiles_close")
    # tree_cloud = o3d.t.io.read_point_cloud(os.path.join(out_trees, "trees_" + tile_name))
    # tree_cloud = tree_cloud.voxel_down_sample(voxel_size=0.03)

    # kd_tree_redivision_single(os.path.basename(tile_name), understory_cloud, tree_cloud)

    
    

    # post_process_pc()
    return


if __name__ == "__main__":
    main()
