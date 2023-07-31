import os
import open3d as o3d
import numpy as np
import argparse
import pandas as pd

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


def trees_test_train_split(bbox_extent, pc_folder, odir):
    bbox = pd.read_csv(bbox_extent)
    bb_x_min = bbox['x_min'].values[0]
    bb_x_max = bbox['x_max'].values[0]
    bb_y_min = bbox['y_min'].values[0]
    bb_y_max = bbox['y_max'].values[0]


    test_x_max = bb_x_min + 1/5*(bb_x_max - bb_x_min)
    val_x_max = test_x_max + 1/5*(bb_x_max - bb_x_min)

    train_odir = os.path.join(odir, "train")
    if not os.path.exists(train_odir):
        os.mkdir(train_odir)
    val_odir = os.path.join(odir, "val")
    if not os.path.exists(val_odir):
        os.mkdir(val_odir)
    test_odir = os.path.join(odir, "test")
    if not os.path.exists(test_odir):
        os.mkdir(test_odir)
    

    # Make mapping of tree file to unique number 
    filenames = [f for f in os.listdir(pc_folder) if f[-3:] == 'ply']

    # plan: divide all tiles of 155 * 124 into 10 and 8 slices independently

    # 20 % test
    # 20 % val
    # 60 % train
    # any tree with at least 5 meters of bbox over val is also assigned to val
    # for train/val just put wherever center is

    for i, filename in enumerate(filenames):
        pcl = o3d.io.read_point_cloud(os.path.join(pc_folder, filename))
        pcl_bbox = pcl.get_axis_aligned_bounding_box()

        bbox_min = pcl_bbox.get_min_bound()
        bbox_max = pcl_bbox.get_max_bound()

        bbox_center = bbox_min + (bbox_max - bbox_min)/2

        if bbox_center[0] < test_x_max:
            out_path = os.path.join(test_odir, filename)
        elif bbox_center[0] < val_x_max or bbox_min[0] < (val_x_max - 5):
            out_path = os.path.join(val_odir, filename)
        else:
            out_path = os.path.join(train_odir, filename)

        o3d.io.write_point_cloud(out_path, pcl)




def main():
    DATA_DIR = "/home/wcherlet/data/singletrees/Wytham2015"
    bbox_extent = os.path.join(DATA_DIR, "extent.csv")
    pc_folder = os.path.join(DATA_DIR, "trees")

    odir = os.path.join(DATA_DIR, "trees_train_val_test")
    if not os.path.exists(odir):
        os.mkdir(odir)

    ds_ddir = os.path.join(pc_folder, "vis_ds")

    # down_sample(pc_folder)

    trees_test_train_split(bbox_extent, pc_folder, odir)


    return


if __name__ == "__main__":
    main()
