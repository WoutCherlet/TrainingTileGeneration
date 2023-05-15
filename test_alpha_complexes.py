import argparse
import os
import time
import glob

import numpy as np
import open3d as o3d


def alpha_complex_o3d(pc, alpha):
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pc, alpha)
    return mesh

def test_decimation(pc, alpha):
    t = time.process_time()
    mesh = alpha_complex_o3d(pc, alpha)
    elapsed_time = time.process_time() - t
    print(f"Alpha complex calculation using open3d took {elapsed_time} seconds")

    n_triangles = len(mesh.triangles)
    print(f"Number of triangles: {n_triangles}")

    t = time.process_time()
    decimated_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(n_triangles//20))
    elapsed_time = time.process_time() - t
    print(f"Decimation took {elapsed_time} seconds")

    decimated_mesh.translate(np.array([10,0,0]))

    decimated_mesh2 = mesh.simplify_quadric_decimation(target_number_of_triangles=int(n_triangles//50))
    decimated_mesh2.translate(np.array([20,0,0]))

    decimated_mesh3 = mesh.simplify_quadric_decimation(target_number_of_triangles=int(n_triangles//100))
    decimated_mesh3.translate(np.array([30,0,0]))

    n_triangles = len(decimated_mesh.triangles)
    print(f"Number of triangles after decimation: {n_triangles}")

    o3d.visualization.draw_geometries([mesh, decimated_mesh, decimated_mesh2, decimated_mesh3], mesh_show_back_face=True, mesh_show_wireframe=True)

def place_trees_aligned(pcs, alpha=0.5, decimation_factor=100, min_n_triangles=2500):
    goal_x = 0
    meshes = []
    for pc in pcs:
        # create mesh
        mesh = alpha_complex_o3d(pc, alpha)
        #decimate to 1/decimation_factor number of triangles
        decimated_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max(int(len(mesh.triangles)//decimation_factor), min_n_triangles))
        print(f"Number of triangles left after decimation: {len(decimated_mesh.triangles)}")
        # get axis-aligned bounding box
        aligned_bbox = decimated_mesh.get_axis_aligned_bounding_box()
        
        bbox_min_x, bbox_min_y, bbox_min_z = aligned_bbox.get_min_bound()

        shift = [goal_x-bbox_min_x, -bbox_min_y, -bbox_min_z]

        decimated_mesh.translate(shift)
        meshes.append(decimated_mesh)
        
        goal_x += aligned_bbox.get_max_bound()[0] - bbox_min_x

    o3d.visualization.draw_geometries(meshes, mesh_show_back_face=True, mesh_show_wireframe=True)



def read_plys(pc_dir):
    pcs = []
    for file in sorted(glob.glob(os.path.join(pc_dir, "*.ply"))):
        pc = o3d.io.read_point_cloud(file)
        pcs.append(pc)
    return pcs


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-p", "--pointcloud", required=True)
    # parser.add_argument("-d", "--pointcloud_directory", required=True)

    # args = parser.parse_args()

    # if not os.path.exists(args.pointcloud):
    #     print(f"Couldn't find pc at {args.pointcloud}!")
    #     return
    # alpha = 0.5
    # o3d_pc = o3d.io.read_point_cloud(args.pointcloud)
    # test_decimation(o3d_pc, alpha)


    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--pointcloud_directory", required=True)

    args = parser.parse_args()

    if not os.path.exists(args.pointcloud_directory):
        print(f"Couldn't reaad path {args.pointcloud_directory}!")
        return
    
    pcs = read_plys(args.pointcloud_directory)

    place_trees_aligned(pcs)

    return

    

if __name__ == "__main__":
    main()
