import os
import open3d as o3d
import numpy as np
import argparse
import pandas as pd

def read_pointclouds(folder):
    out = {}

    for file in os.listdir(folder):
        # read like this to delete custom attributes
        pc = o3d.io.read_point_cloud(os.path.join(folder, file))
        # remove colors if present, otherwise merge no work
        pc.colors = o3d.utility.Vector3dVector()
        pc = o3d.t.geometry.PointCloud.from_legacy(pc)
        out[file] = pc

    return out

def merge_all(pointclouds):
    pointcloud = pointclouds[0]
    for pc in pointclouds[1:]:
        pointcloud += pc
    return pointcloud


def down_sample(pc_folder):
    filenames = [f for f in os.listdir(pc_folder) if f[-3:] == 'ply']

    odir = os.path.join(pc_folder, "vis_ds")
    if not os.path.exists(odir):
        os.mkdir(odir)

    for i, filename in enumerate(filenames):
        pcl = o3d.io.read_point_cloud(os.path.join(pc_folder, filename))

        pcl = pcl.voxel_down_sample(voxel_size=0.20)

        out_path = os.path.join(odir, filename)

        o3d.io.write_point_cloud(out_path, pcl)


def trees_test_train_split(tree_tiles_dict, understory_tiles_dict, trees_folder, odir):
    pointclouds = list(tree_tiles_dict.values())
    pointclouds += list(understory_tiles_dict.values())

    merged_pointcloud = merge_all(pointclouds)

    max_bound = merged_pointcloud.get_max_bound().numpy()
    min_bound = merged_pointcloud.get_min_bound().numpy()

    bound_x_min = min_bound[0]
    bound_x_max = max_bound[0]


    test_x_max = bound_x_min + 1/5*(bound_x_max - bound_x_min)
    val_x_max = test_x_max + 1/5*(bound_x_max - bound_x_min)

    train_odir = os.path.join(odir, "trees", "train")
    if not os.path.exists(train_odir):
        os.makedirs(train_odir)
    val_odir = os.path.join(odir, "trees", "val")
    if not os.path.exists(val_odir):
        os.makedirs(val_odir)
    test_odir = os.path.join(odir, "trees", "test")
    if not os.path.exists(test_odir):
        os.makedirs(test_odir)

    
    # Make mapping of tree file to unique number 
    filenames = [f for f in os.listdir(trees_folder) if f[-3:] == 'ply']

    # Divide wytham plot into 80 tiles
    # 10 across 135m. length x-axis
    # 8 across 88m. length y-axis

    # train/val/test division across x-axis

    # 20 % test
    # 20 % val
    # 60 % train
    # any tree with at least 5 meters of bbox over val is also assigned to val
    # for train/val just put wherever center is

    test_trees = []
    val_trees = []
    train_trees = []

    print("dividing trees")

    for i, filename in enumerate(filenames):
        pcl = o3d.t.io.read_point_cloud(os.path.join(trees_folder, filename))

        # add labels

        pcl.point.semantic = o3d.core.Tensor(np.ones(len(pcl.point.positions), dtype=np.int32)[:,None])
        pcl.point.instance = o3d.core.Tensor((i+1)*np.ones(len(pcl.point.positions), dtype=np.int32)[:,None])

        pcl_bbox = pcl.get_axis_aligned_bounding_box()

        bbox_min = pcl_bbox.min_bound
        bbox_max = pcl_bbox.max_bound

        bbox_center = bbox_min + (bbox_max - bbox_min)/2

        if bbox_center[0] < test_x_max:
            out_path = os.path.join(test_odir, filename)
            test_trees.append(pcl)
        elif bbox_center[0] < val_x_max or bbox_min[0] < (val_x_max - 5):
            out_path = os.path.join(val_odir, filename)
            val_trees.append(pcl)
        else:
            out_path = os.path.join(train_odir, filename)
            train_trees.append(pcl)

        o3d.t.io.write_point_cloud(out_path, pcl)

    
    # merge all understory tiles that overlap with test, val and train areas
    test_tiles = []
    val_tiles = []
    train_tiles = []

    print("dividing understory tiles")
    
    for tile in understory_tiles_dict:
        pc = understory_tiles_dict[tile]
        min_bound_x, _, _ = pc.get_min_bound().numpy()
        max_bound_x, _, _ = pc.get_max_bound().numpy()

        if min_bound_x < test_x_max:
            test_tiles.append(pc)
        if min_bound_x < val_x_max and max_bound_x > test_x_max:
            val_tiles.append(pc)
        if max_bound_x > val_x_max:
            train_tiles.append(pc)

    test_merged = merge_all(test_tiles)
    val_merged = merge_all(val_tiles)
    train_merged = merge_all(train_tiles)

    # add terrain labels: semantic is 0, instance = -1

    test_merged.point.semantic = o3d.core.Tensor(np.zeros(len(test_merged.point.positions), dtype=np.int32)[:,None])
    test_merged.point.instance = o3d.core.Tensor((-1)*np.ones(len(test_merged.point.positions), dtype=np.int32)[:,None])
    val_merged.point.semantic = o3d.core.Tensor(np.zeros(len(val_merged.point.positions), dtype=np.int32)[:,None])
    val_merged.point.instance = o3d.core.Tensor((-1)*np.ones(len(val_merged.point.positions), dtype=np.int32)[:,None])
    train_merged.point.semantic = o3d.core.Tensor(np.zeros(len(train_merged.point.positions), dtype=np.int32)[:,None])
    train_merged.point.instance = o3d.core.Tensor((-1)*np.ones(len(train_merged.point.positions), dtype=np.int32)[:,None])

    print("merging trees and understory")

    # merge trees
    test_trees_pc = merge_all(test_trees)
    val_trees_pc = merge_all(val_trees)
    train_trees_pc = merge_all(train_trees)

    test_plot_pc = test_trees_pc + test_merged
    val_plot_pc = val_trees_pc + val_merged
    train_plot_pc = train_trees_pc + train_merged


    # slice merged test, val and train pointclouds into actual sizes
    test_max_bound = max_bound.copy()
    test_max_bound[0]  = test_x_max

    val_max_bound = max_bound.copy()
    val_max_bound[0] = val_x_max
    val_min_bound = min_bound.copy()
    val_min_bound[0] = test_x_max

    test_min_bound = min_bound.copy()
    test_min_bound[0] = val_x_max

    test_bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound = min_bound, max_bound = test_max_bound)
    val_bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound = val_min_bound, max_bound = val_max_bound)
    train_bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound = test_min_bound, max_bound = max_bound)

    test_sliced = test_plot_pc.crop(test_bbox)
    val_sliced = val_plot_pc.crop(val_bbox)
    train_sliced = train_plot_pc.crop(train_bbox)

    # write out test, val and train plots as temp
    o3d.t.io.write_point_cloud(os.path.join(odir, "test_merged.ply"), test_sliced)
    o3d.t.io.write_point_cloud(os.path.join(odir, "val_merged.ply"), val_sliced)
    o3d.t.io.write_point_cloud(os.path.join(odir, "train_merged.ply"), train_sliced)

    return


def tile_area(merged_area, x_n, y_n, odir):

    OVERLAP = 5

    min_bound = merged_area.get_min_bound().numpy()
    max_bound = merged_area.get_max_bound().numpy()


    x_tile_size = (max_bound[0] - min_bound[0] - OVERLAP) / x_n + OVERLAP
    y_tile_size = (max_bound[1] - min_bound[1] - OVERLAP) / y_n + OVERLAP

    print(f"Tile sizes: {x_tile_size}, {y_tile_size}")

    tile_n = 0

    all_tiles = []
    
    for i in range(x_n):
        for j in range(y_n):
            tile_min_bound = min_bound.copy()
            tile_min_bound[0] += i*(x_tile_size - OVERLAP)
            tile_min_bound[1] += j*(y_tile_size - OVERLAP)

            tile_max_bound = max_bound.copy()
            tile_max_bound[0] = tile_min_bound[0] + x_tile_size
            tile_max_bound[1] = tile_min_bound[1] + y_tile_size

            tile_bbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound = tile_min_bound, max_bound = tile_max_bound)
            tile_pc = merged_area.crop(tile_bbox)

            o3d.t.io.write_point_cloud(os.path.join(odir, f"Wytham_Tile{tile_n}.ply"), tile_pc)
            tile_n += 1

            # TODO: temp: shift and save
            # tile_pc = tile_pc.translate(np.array([i*7, j*7, 0]))
            # all_tiles.append(tile_pc)

    # TODO: TEMP
    # o3d.visualization.draw_geometries(all_tiles)
    pass

def tile_wytham(merged_area_dir):

    # training area

    odir = os.path.join(merged_area_dir, "tiles", "train")
    if not os.path.exists(odir):
        os.makedirs(odir)
    train_pc = o3d.t.io.read_point_cloud(os.path.join(merged_area_dir, "train_merged.ply"))

    tile_area(train_pc, x_n=6, y_n=11, odir=odir)

    # val area

    odir = os.path.join(merged_area_dir, "tiles", "val")
    if not os.path.exists(odir):
        os.makedirs(odir)
    validation_pc = o3d.t.io.read_point_cloud(os.path.join(merged_area_dir, "val_merged.ply"))

    tile_area(validation_pc, x_n=2, y_n=11, odir=odir)

    # test area

    odir = os.path.join(merged_area_dir, "tiles", "test")
    if not os.path.exists(odir):
        os.makedirs(odir)
    test_pc = o3d.t.io.read_point_cloud(os.path.join(merged_area_dir, "test_merged.ply"))

    tile_area(test_pc, x_n=2, y_n=11, odir=odir)


def main():
    DATA_DIR = "/home/wcherlet/data/Wytham_cleaned/"
    trees_folder = os.path.join(DATA_DIR, "trees")

    TILE_DIR = "/home/wcherlet/data/Wytham_cleaned/seperated"

    TREES_DIR = os.path.join(TILE_DIR, "trees_kd")
    UNDERSTORY_DIR = os.path.join(TILE_DIR, "understory_kd")

    tree_tiles_dict = read_pointclouds(TREES_DIR)
    understory_tiles_dict = read_pointclouds(UNDERSTORY_DIR)

    odir = os.path.join(DATA_DIR, "Wytham_train_split")
    if not os.path.exists(odir):
        os.mkdir(odir)

    # trees_test_train_split(tree_tiles_dict, understory_tiles_dict, trees_folder, odir)

    tile_wytham(odir)


    return


if __name__ == "__main__":
    main()
