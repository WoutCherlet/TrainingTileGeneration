import argparse
import os
import time
import glob

import numpy as np
import open3d as o3d

def pc_2_alphacomplex(pc_dir, out_dir, alpha=0.5, decimation_factor=100, min_n_triangles=2500):
    pcs = []
    for file in sorted(glob.glob(os.path.join(pc_dir, "*.ply"))):
        pc = o3d.io.read_point_cloud(file)
        name = os.path.basename(file)[:-4]
        pcs.append((name, pc))

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for name, pc in pcs:
        print(f"Processing pc {name}")
        # create mesh
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pc, alpha)
        #decimate to 1/decimation_factor number of triangles
        decimated_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max(int(len(mesh.triangles)//decimation_factor), min_n_triangles))
        print(f"Number of triangles left after decimation: {len(decimated_mesh.triangles)}")

        o3d.io.write_triangle_mesh(os.path.join(out_dir, name+f"_alphacomplex_{alpha}.ply"), decimated_mesh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--pointcloud_directory", required=True)
    parser.add_argument("-o", "--output_directory", default=None)

    args = parser.parse_args()

    if not os.path.exists(args.pointcloud_directory):
        print(f"Couldn't read input dir {args.pointcloud_directory}!")
        return
    
    if args.output_directory is not None:
        if not os.path.exists(args.output_directory):
            print(f"Couldn't read output dir {args.output_directory}!")
            return
        out_dir = args.output_directory
    else:
        out_dir = os.path.join(args.pointcloud_directory, "alpha_complexes")
    
    pc_2_alphacomplex(args.pointcloud_directory, out_dir)

    return

    

if __name__ == "__main__":
    main()