import argparse
import os
import glob
import random
import math as m

import numpy as np
import open3d as o3d
import trimesh

def read_trees(mesh_dir, pc_dir, alpha=None):
    trees = {}
    for file in sorted(glob.glob(os.path.join(pc_dir, "*.ply"))):
        pc = o3d.io.read_point_cloud(file)
        name = os.path.basename(file)[:-4]
        # find file with similar name in mesh_dir
        if isinstance(alpha, float):
            search_term = os.path.join(mesh_dir, name + f"_alphacomplex_{str(alpha)}.ply")
        else:
            search_term = os.path.join(mesh_dir, name + "_alphacomplex_*.ply" )
        results = sorted(glob.glob(search_term))
        if len(results) == 0:
            print(f"Didn't find alpha complex file for tree {name}, skipping")
            continue
        mesh_file = results[0]
        if len(results) > 1:
            print(f"Found multiple alpha complexes for tree {name}, using {mesh_file}")
        o3d_mesh = o3d.io.read_triangle_mesh(mesh_file)
        tri_mesh = trimesh.load(mesh_file)
        trees[name] = (pc, o3d_mesh, tri_mesh)
    return trees

def place_tree_in_plot(name, tree_mesh, plot_mesh, collision_manager):

    # start by placing tree at edge of plot
    translation = np.zeros((4,4), dtype=float)

    plot_bounds = plot_mesh.bounds
    max_x_plot = plot_bounds[1][0]
    min_z_plot = plot_bounds[0][2]

    min_x_tree, min_y_tree, min_z_tree = tree_mesh.bounds[0]

    bbox_transform = trimesh.transformations.translation_matrix(np.array([max_x_plot-min_x_tree, -min_y_tree, min_z_plot-min_z_tree]))

    tree_mesh.apply_transform(bbox_transform)
    translation += bbox_transform

    placed = False
    distance_buffer = 0.05
    x_dir = True
    i = 0
    max_iterations = 100

    while not placed:
        # get distance of mesh to plot
        min_distance, closest_name, distance_data = collision_manager.min_distance_single(tree_mesh, return_name=True, return_data=True)

        if min_distance < distance_buffer:
            print(f"placed after { i } iterations")
            placed = True
        else:
            # TODO: moving point to bbox center will most likely give more organic results, when moving closest points together we don't get a lot of overlap as gets stuck quickly getting two external points close together
            # using bbox centers will give some randomness to it
            closest_point_plot = distance_data.point(closest_name)[:2]

            # external is name of current tree in distance data object
            closest_point_tree = distance_data.point("__external")[:2]

            offsets = closest_point_plot - closest_point_tree

            sign_x = m.copysign(1, offsets[0])
            sign_y = m.copysign(1, offsets[1])

            trans = [sign_x*(abs(offsets[0])-distance_buffer/2), sign_y*(abs(offsets[1])-distance_buffer/2), 0]

            bbox_transform = trimesh.transformations.translation_matrix(trans)
            tree_mesh.apply_transform(bbox_transform)
            translation += bbox_transform
        
        if i > max_iterations:
            print(f"placed after max iterations { max_iterations}")
            placed = True
        
        i += 1
        x_dir = not x_dir
    
    collision_manager.add_object(name, tree_mesh)
    plot_mesh += tree_mesh

    return plot_mesh, translation



def assemble_trees(trees, n_trees=10):
    transforms = []

    trees_list = list(trees.keys())

    tri_mesh_plot = None

    collision_manager = trimesh.collision.CollisionManager()

    for i in range(n_trees):
        o3d_transform = np.zeros((4,4), dtype=float)

        # pick random tree
        # name = random.choice(trees_list)
        name = trees_list[i]
        trees_list.remove(name)

        _, _, tri_mesh = trees[name]

        # generate random rotation around z axis
        rot_angle = m.radians(random.randrange(360))
        rot_matrix = np.identity(4, dtype=float)
        rot_matrix[:2,:2] = [[m.cos(rot_angle), -m.sin(rot_angle)], [m.sin(rot_angle),m.cos(rot_angle)]]

        # save rotation and apply to trimesh mesh
        o3d_transform += rot_matrix
        tri_mesh.apply_transform(rot_matrix)

        if i == 0:
            # start plot at 0,0,0
            min_x_mesh, min_y_mesh, min_z_mesh = tri_mesh.bounds[0]
            origin_translation = trimesh.transformations.translation_matrix(np.array([-min_x_mesh, -min_y_mesh, -min_z_mesh]))
            tri_mesh.apply_transform(origin_translation)
            tri_mesh_plot = tri_mesh
            collision_manager.add_object(name, tri_mesh_plot)
        else:
            tri_mesh_plot, translation = place_tree_in_plot(name, tri_mesh, tri_mesh_plot, collision_manager)
            o3d_transform += translation

        # TODO: save transform here and apply later to pointclouds to get final tile pointcloud
    
    return tri_mesh_plot



def generate_tile(mesh_dir, pc_dir, out_dir, alpha=None):
    trees = read_trees(mesh_dir, pc_dir, alpha=alpha)

    plot = assemble_trees(trees, n_trees=3)

    plot.show()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--mesh_directory", required=True)
    parser.add_argument("-p", "--pointcloud_directory", required=True)
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
        out_dir = os.path.join(args.pointcloud_directory, "tiles")
    
    generate_tile(args.mesh_directory, args.pointcloud_directory, out_dir)

    return

    

if __name__ == "__main__":
    main()