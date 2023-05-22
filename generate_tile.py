import argparse
import os
import glob
import random
import math as m

import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
from perlin_numpy import generate_fractal_noise_2d, generate_perlin_noise_2d

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


def place_tree_in_line(name, tree_mesh, plot_mesh, collision_manager, trees):
    # start by placing tree at edge of plot
    translation = np.zeros((4,4), dtype=float)

    plot_bounds = plot_mesh.bounds
    max_x_plot = plot_bounds[1][0]
    min_z_plot = plot_bounds[0][2]

    min_x_tree, min_y_tree, min_z_tree = tree_mesh.bounds[0]

    bbox_transform = trimesh.transformations.translation_matrix(np.array([max_x_plot-min_x_tree, -min_y_tree, min_z_plot-min_z_tree]))

    tree_mesh.apply_transform(bbox_transform)
    translation += bbox_transform

    # move tree as close as possible to rest of plot
    placed = False
    distance_buffer = 0.05
    i = 1
    max_iterations = 100

    while not placed:
        # get distance of mesh to plot
        min_distance, closest_name = collision_manager.min_distance_single(tree_mesh, return_name=True)

        if min_distance < distance_buffer:
            print(f"placed after { i } iterations")
            placed = True
        else:
            # move tree in direction of closest bbox center by min_distance with little buffer
            bbox_xy_center_current = (tree_mesh.bounds[1][:2] + tree_mesh.bounds[0][:2]) / 2
            _, _, closest_tree = trees[closest_name]
            bbox_xy_center_closest = (closest_tree.bounds[1][:2] - closest_tree.bounds[0][:2]) / 2

            direction_vector = bbox_xy_center_closest - bbox_xy_center_current
            # unit vector scaled with min_distance ensures there is never a collision, as we move within sphere with radius min_distance
            unit_vector = direction_vector / np.linalg.norm(direction_vector)

            # add noise to x and y and renormalize
            noisy_vector = [unit_vector[0] + random.uniform(-0.5, 0.5), unit_vector[1] + random.uniform(-0.5, 0.5)]
            noisy_unit_vector = noisy_vector / np.linalg.norm(noisy_vector)

            trans_distance = min_distance - distance_buffer / 2

            trans = [noisy_unit_vector[0]*trans_distance, noisy_unit_vector[1]*trans_distance, 0]

            bbox_transform = trimesh.transformations.translation_matrix(trans)
            tree_mesh.apply_transform(bbox_transform)
            translation += bbox_transform
        
        if i >= max_iterations:
            print(f"placed after max iterations { max_iterations}")
            placed = True
        
        i += 1

    plot_mesh += tree_mesh

    return plot_mesh, translation

def assemble_trees_line(trees, n_trees=9):
    transforms = []

    trees_list = list(trees.keys())

    tri_mesh_plot = None

    collision_manager = trimesh.collision.CollisionManager()

    n_row = m.floor(m.sqrt(n_trees))

    for i in range(n_trees):
        
        # pick random tree
        # name = random.choice(trees_list)
        name = trees_list[i]
        print(f"Placing tree {name}")
        trees_list.remove(name)
        _, _, tri_mesh = trees[name]

        # save all transforms for this tree to single matrix
        o3d_transform = np.zeros((4,4), dtype=float)

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
            tri_mesh_plot, translation = place_tree_in_line(name, tri_mesh, tri_mesh_plot, collision_manager, trees)
            o3d_transform += translation

        # TODO: save transform here and apply later to pointclouds to get final tile pointcloud
    
    return tri_mesh_plot


def place_tree_in_grid(name, tree_mesh, collision_manager_plot, collision_manager_row, trees, max_x_row, max_y_plot, debug=False):
    # start by placing tree at edge of plot
    total_translation = np.array([0.0,0.0,0.0])

    min_x_tree, min_y_tree, min_z_tree = tree_mesh.bounds[0]

    initial_translation = np.array([max_x_row-min_x_tree, max_y_plot-min_y_tree, -min_z_tree])
    bbox_transform = trimesh.transformations.translation_matrix(initial_translation)

    tree_mesh.apply_transform(bbox_transform)
    total_translation += initial_translation

    # move tree as close as possible to rest of plot
    placed = False
    if debug:
        placed = True
    distance_buffer = 0.05
    i = 1
    max_iterations = 100

    while not placed:
        # get distance of mesh to plot
        if collision_manager_plot is not None:
            min_distance_plot, closest_name_plot = collision_manager_plot.min_distance_single(tree_mesh, return_name=True)
        else:
            min_distance_plot = 1e9 # cant use inf because of mult with 0
            closest_name_plot = None

        # get distance of mesh to row
        if max_x_row != 0:
            min_distance_row, closest_name_row = collision_manager_row.min_distance_single(tree_mesh, return_name=True)
        else:
            min_distance_row = 1e9 # cant use inf because of mult with 0
            closest_name_row = None

        min_distance = min(min_distance_plot, min_distance_row)

        if min_distance < distance_buffer:
            print(f"placed after { i } iterations")
            placed = True
        else:
            # move tree in direction of linear combination of closest bbox center of plot and row by min_distance with little buffer
            bbox_xy_center_current = (tree_mesh.bounds[1][:2] + tree_mesh.bounds[0][:2]) / 2

            if closest_name_plot is not None:
                _, _, closest_tree_plot = trees[closest_name_plot]
                bbox_xy_center_closest_plot = (closest_tree_plot.bounds[1][:2] - closest_tree_plot.bounds[0][:2]) / 2
                direction_vector_plot = np.array(bbox_xy_center_closest_plot - bbox_xy_center_current)
            else:
                direction_vector_plot = np.array([0,0])

            if closest_name_row is not None:
                _, _, closest_tree_row= trees[closest_name_row]
                bbox_xy_center_closest_row = (closest_tree_row.bounds[1][:2] - closest_tree_row.bounds[0][:2]) / 2
                direction_vector_row = np.array(bbox_xy_center_closest_row - bbox_xy_center_current)
            else:
                direction_vector_row = np.array([0,0])
                
            # weighted by closest distance
            direction_vector = min_distance_plot**2 * direction_vector_plot + min_distance_row * direction_vector_row
            unit_vector = direction_vector / np.linalg.norm(direction_vector)

            # add noise to x and y and renormalize
            noisy_vector = [unit_vector[0] + random.uniform(-0.5, 0.5), unit_vector[1] + random.uniform(-0.5, 0.5)]
            noisy_unit_vector = noisy_vector / np.linalg.norm(noisy_vector)

            trans_distance = min_distance - distance_buffer / 2

            trans = [noisy_unit_vector[0]*trans_distance, noisy_unit_vector[1]*trans_distance, 0]

            bbox_transform = trimesh.transformations.translation_matrix(trans)
            tree_mesh.apply_transform(bbox_transform)
            total_translation += trans
        
        if i >= max_iterations:
            print(f"placed after max iterations { max_iterations}")
            placed = True
        
        i += 1

    max_x_row = tree_mesh.bounds[1][0]
    collision_manager_row.add_object(name, tree_mesh)

    translation_matrix = trimesh.transformations.translation_matrix(total_translation)

    return tree_mesh, translation_matrix, max_x_row

def assemble_trees_grid(trees, n_trees=9, debug=False):

    if debug:
        print("Debug is True, going through trees in order, not rotating and placing at bbox edge + drawing bboxs")

    transforms = {}

    trees_list = list(trees.keys())

    tri_mesh_plot = None

    n_row = m.floor(m.sqrt(n_trees))

    for i in range(n_trees):
        # pick random tree
        if debug:
            name = trees_list[i]
        else:
            name = random.choice(trees_list)
            trees_list.remove(name)
        
        _, _, tri_mesh = trees[name]

        # save all transforms for this tree to single matrix
        o3d_transform = np.identity(4, dtype=float)

        if not debug:
            # generate random rotation around z axis
            rot_angle = m.radians(random.randrange(360))
            rot_matrix = np.identity(4, dtype=float)
            rot_matrix[:2,:2] = [[m.cos(rot_angle), -m.sin(rot_angle)], [m.sin(rot_angle),m.cos(rot_angle)]]

            # save rotation and apply to trimesh mesh
            o3d_transform = np.matmul(rot_matrix, o3d_transform)
            tri_mesh.apply_transform(rot_matrix)


        if i == 0:
            # start plot at 0,0,0
            print(f"Placing tree {name}")
            min_x_mesh, min_y_mesh, min_z_mesh = tri_mesh.bounds[0]
            origin_translation = trimesh.transformations.translation_matrix(np.array([-min_x_mesh, -min_y_mesh, -min_z_mesh]))
            tri_mesh.apply_transform(origin_translation)
            tri_mesh_plot = tri_mesh
            if debug:
                tri_mesh_plot += tri_mesh.bounding_box
            o3d_transform = np.matmul(origin_translation, o3d_transform)

            # init plot metrics and collision managers
            collision_manager_row = trimesh.collision.CollisionManager()
            collision_manager_row.add_object(name, tri_mesh)
            collision_manager_plot = None
            max_x_row = tri_mesh.bounds[1][0]
            max_y_plot = 0
            collision_meshes = {}
            collision_meshes[name] = tri_mesh
        else:
            new_row = i % n_row == 0
            # on new row: update plot metrics and collision managers
            if new_row:
                print(f"starting new row (i={i})")
                max_x_row = 0
                max_y_plot = tri_mesh_plot.bounds[1][1]
                if collision_manager_plot is None:
                    collision_manager_plot = trimesh.collision.CollisionManager()
                row_objects = collision_manager_row._objs.copy()
                for object in row_objects:
                    collision_manager_plot.add_object(object, collision_meshes[object])
                    collision_manager_row.remove_object(object)
            
            print(f"Placing tree {name}")
            
            tree_mesh, translation, max_x_row = place_tree_in_grid(name, tri_mesh, collision_manager_plot, collision_manager_row, trees, max_x_row, max_y_plot, debug=debug)
            collision_meshes[name] = tree_mesh
            tri_mesh_plot += tree_mesh
            if debug:
                tri_mesh_plot += tree_mesh.bounding_box
            o3d_transform = np.matmul(translation, o3d_transform)

        # save_transforms
        transforms[name] = o3d_transform
    
    return tri_mesh_plot, transforms

def add_terrain_flat(plot_cloud, height=0.0, points_per_meter = 10):

    # get dimension of plot cloud
    max_x, max_y, _ = plot_cloud.get_max_bound()
    min_x, min_y, _ = plot_cloud.get_min_bound()

    nx = round((max_x - min_x) * points_per_meter)
    ny = round((max_y - min_y) * points_per_meter)

    x = np.linspace(min_x, max_x, num = nx)
    y = np.linspace(min_y, max_y, num = ny)
    xv, yv = np.meshgrid(x, y)


    points_xy = np.array([xv.flatten(), yv.flatten()]).T
    z_arr = np.full((len(points_xy),1), height)
    points_3d = np.hstack((points_xy, z_arr))
    vector_3d = o3d.utility.Vector3dVector(points_3d)

    terrain_cloud = o3d.geometry.PointCloud(vector_3d)

    return terrain_cloud

def add_terrain_perlin_noise(plot_cloud, height=0.0, points_per_meter = 10):
    # get dimension of plot cloud
    max_x, max_y, _ = plot_cloud.get_max_bound()
    min_x, min_y, _ = plot_cloud.get_min_bound()

    nx = round((max_x - min_x) * points_per_meter)
    ny = round((max_y - min_y) * points_per_meter)

    res = 4

    nx = nx - (nx % res)
    ny = ny - (ny % res)

    perlin_noise = generate_perlin_noise_2d((ny, nx), (res, res))

    x = np.linspace(min_x, max_x, num = nx)
    y = np.linspace(min_y, max_y, num = ny)
    xv, yv = np.meshgrid(x, y)


    points_xy = np.array([xv.flatten(), yv.flatten()]).T
    z_arr = perlin_noise.flatten()
    points_3d = np.column_stack((points_xy, z_arr))

    plt.imshow(perlin_noise, cmap='gray', interpolation='lanczos')
    plt.colorbar()
    plt.show()

    vector_3d = o3d.utility.Vector3dVector(points_3d)

    terrain_cloud = o3d.geometry.PointCloud(vector_3d)

    return terrain_cloud


def generate_tile(mesh_dir, pc_dir, out_dir, alpha=None):
    trees = read_trees(mesh_dir, pc_dir, alpha=alpha)

    print(f"Read {len(trees)} trees")

    plot, transforms = assemble_trees_grid(trees, n_trees=4, debug=True)
    
    # apply derived transforms to pointcloud 

    merged_plot = None
    for name in transforms:
        # apply transforms to open3d and merge

        pc, _, _ = trees[name]
        transform = transforms[name]
        # transform_tensor = o3d.core.Tensor(transform)
        pc = pc.transform(transform)

        # ugly code but for some reason o3d errors when appending to empty pointcloud
        if merged_plot is None:
            # copy constructor
            merged_plot = o3d.geometry.PointCloud(pc)
        else:
            merged_plot += pc
        continue

    # add flat terrain
    # TODO: noisy terrain
    # TODO: mesh of terrain for collision detection
    terrain_cloud = add_terrain_perlin_noise(merged_plot)
    merged_plot += terrain_cloud

    
    plot.show()

    o3d.visualization.draw_geometries([merged_plot])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pointcloud_directory", required=True)
    parser.add_argument("-d", "--mesh_directory", default=None)
    parser.add_argument("-o", "--output_directory", default=None)

    args = parser.parse_args()

    if not os.path.exists(args.pointcloud_directory):
        print(f"Couldn't read input dir {args.pointcloud_directory}!")
        return
    
    if args.mesh_directory is None:
        if not os.path.exists(os.path.join(args.pointcloud_directory, "alpha_complexes")):
            print(f"No mesh directory provided and not found at {os.path.join(args.pointcloud_directory, 'alpha_complexes')}, exiting.")
            return
        else:
            args.mesh_directory = os.path.join(args.pointcloud_directory, "alpha_complexes")
    
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