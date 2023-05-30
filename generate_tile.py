import argparse
import os
import glob
import random
import math as m

import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from perlin_numpy import generate_fractal_noise_2d

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


def get_trunk_location(pointcloud):
    # get axis aligned bounding box
    aaligned_bbox = pointcloud.get_axis_aligned_bounding_box()

    # slice bottom meter of bounding box
    min_bound_bbox = aaligned_bbox.get_min_bound()
    max_bound_bbox = aaligned_bbox.get_max_bound()
    max_bound_bbox[2] = min_bound_bbox[2] + 1
    aaligned_bbox.max_bound = max_bound_bbox

    # crop pointcloud to bottom bbox
    cropped_pointcloud = pointcloud.crop(aaligned_bbox)

    cropped_bbox = cropped_pointcloud.get_axis_aligned_bounding_box()

    center_x, center_y = cropped_bbox.get_center()[:2]
    # Temporary: get convex hull center
    return [center_x, center_y, min_bound_bbox[2]]

def get_trunk_convex_hull(pointcloud, slice_height=0.5):  
    # get axis aligned bounding box
    aaligned_bbox = pointcloud.get_axis_aligned_bounding_box()

    # slice bottom part of bounding box
    min_bound_bbox = aaligned_bbox.get_min_bound()
    max_bound_bbox = aaligned_bbox.get_max_bound()
    max_bound_bbox[2] = min_bound_bbox[2] + slice_height
    aaligned_bbox.max_bound = max_bound_bbox

    # crop pointcloud to bottom bbox
    cropped_pointcloud = pointcloud.crop(aaligned_bbox)

    points_3d = np.asarray(cropped_pointcloud.points)
    points_projected_2d = points_3d[:,:2]

    hull = ConvexHull(points_projected_2d)
    
    hull_points_3d = points_3d[hull.vertices]
    return hull_points_3d, hull


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

def add_terrain(plot_cloud, trunk_hulls):
    # get dimension of plot cloud
    max_x, max_y, _ = plot_cloud.get_max_bound()
    min_x, min_y, _ = plot_cloud.get_min_bound()

    POINTS_PER_METER = 10 # NOTE: if changing this, will probably need to recalibrate a lot of the other parameters too

    nx = round((max_x - min_x) * POINTS_PER_METER)
    ny = round((max_y - min_y) * POINTS_PER_METER)

    # constants for noise generation
    RES = 4
    LACUNARITY = 2
    OCTAVES = 4

    # get dimensions to generate perlin noise, shape must be multiple of res*lacunarity**(octaves - 1)
    shape_factor = RES*(LACUNARITY**(OCTAVES-1))
    perlin_nx = nx - (nx % shape_factor) + shape_factor
    perlin_ny = ny - (ny % shape_factor) + shape_factor

    perlin_noise = generate_fractal_noise_2d((perlin_ny, perlin_nx), (RES, RES), octaves=OCTAVES, lacunarity=LACUNARITY)
    perlin_noise = perlin_noise[:ny, :nx]
    SCALE = 3
    perlin_noise = perlin_noise * SCALE

    # get influence map and height map of trunks to adapt terrain to trunk heights and locations
    influence_map, height_map = trunk_height_influence_map_convex_circle(min_x, min_y, ny, nx, POINTS_PER_METER, trunk_hulls=trunk_hulls)
    final_xy_map = influence_map * height_map + (np.ones(influence_map.shape, dtype=float) - influence_map) * perlin_noise

    # get xy grid
    x = np.linspace(min_x, max_x, num = nx)
    y = np.linspace(min_y, max_y, num = ny)
    xv, yv = np.meshgrid(x, y)
    
    # apply height map
    points_xy = np.array([xv.flatten(), yv.flatten()]).T
    z_arr = final_xy_map.flatten()
    points_3d = np.column_stack((points_xy, z_arr))

    points_3d = remove_points_inside_hulls(points_3d, trunk_hulls)

    # visualization
    # plt.matshow(final_xy_map, cmap='gray', interpolation='lanczos')
    # plt.colorbar()
    # plt.show()

    # to pointcloud
    vector_3d = o3d.utility.Vector3dVector(points_3d)
    terrain_cloud = o3d.geometry.PointCloud(vector_3d)

    return terrain_cloud


def influence_function(index, total_points):
    x = index/total_points
    # return 1/(1 + (x/(1-x))**2)  # reverse S shape, seems to give too much of "pedestal" kind of form
    # return -(x-1)**3 # simple exponential-like curve 
    # NOTE: use exponential? will never get 0 so no div by 0 errors eg e^-4*x is practically equivalent to above
    return (x-1)**2

def influence_function_dist(distance, max_distance):
    if distance >= max_distance:
        return 0
    x = distance/max_distance
    return (x-1)**2

def trunk_height_influence_map(min_x, min_y, ny, nx, points_per_meter, trunk_locations):
    # NOTE: don't use this, use convex hull function instead

    # create trunk_locations map
    influence_map = np.zeros((ny, nx), dtype= float)
    height_map = np.zeros((ny, nx), dtype=float)

    for tree in trunk_locations:

        center = trunk_locations[tree]
        height = center[2]

        closest_index_x = int(np.round((center[0] - min_x ) * points_per_meter))
        closest_index_y = int(np.round((center[1] - min_y ) * points_per_meter))

        influence_map[closest_index_y, closest_index_x] = 1.0

        height_map[closest_index_y, closest_index_x] = height
        
        # set influence around trunk centers in square form
        TOTAL_POINTS = 50
        for index_offset in range(1, TOTAL_POINTS):
            for x in range(closest_index_x-index_offset, closest_index_x+index_offset+1):
                influence_map[closest_index_y-index_offset, x] = influence_function(index_offset, TOTAL_POINTS)
                influence_map[closest_index_y+index_offset, x] = influence_function(index_offset, TOTAL_POINTS)

                height_map[closest_index_y-index_offset, x] = height
                height_map[closest_index_y+index_offset, x] = height

            for y in range(closest_index_y-index_offset+1, closest_index_y+index_offset):
                influence_map[y, closest_index_x-index_offset] = influence_function(index_offset, TOTAL_POINTS)
                influence_map[y, closest_index_x+index_offset] = influence_function(index_offset, TOTAL_POINTS)

                height_map[y, closest_index_x-index_offset] = height
                height_map[y, closest_index_x+index_offset] = height

    
    # temp: visualize
    # plt.matshow(influence_map, cmap='gray')
    # plt.colorbar()
    # plt.show()

    return influence_map, height_map

def trunk_height_influence_map_convex(min_x, min_y, ny, nx, points_per_meter, trunk_hulls):
    # create trunk influence and height map
    influence_map = np.zeros((ny, nx), dtype= float)
    height_map = np.zeros((ny, nx), dtype=float)

    # used to keep track of influence of seen points on each points height, to do order independent weighted average
    total_past_influence_map = np.zeros((ny, nx), dtype=float)

    for tree in trunk_hulls:
        hull_3d, _ = trunk_hulls[tree]

        # for each point in hull: calculate infuence + set surrounding height
        for point in hull_3d:
            cur_height = point[2]
            
            closest_idx_x = int(np.round((point[0] - min_x ) * points_per_meter))
            closest_idx_y = int(np.round((point[1] - min_y ) * points_per_meter))

            # height = weighted average between old height and current height, based on influence
            height_map[closest_idx_y, closest_idx_x] = (total_past_influence_map[closest_idx_y, closest_idx_x]*height_map[closest_idx_y, closest_idx_x] + 1.0*cur_height) / (total_past_influence_map[closest_idx_y, closest_idx_x] + 1.0)
            influence_map[closest_idx_y, closest_idx_x] = 1.0
            total_past_influence_map[closest_idx_y, closest_idx_x] += 1

            # set influence around trunk centers in square form
            TOTAL_POINTS = 20
            for idx_offset in range(1, TOTAL_POINTS):
                cur_influence = influence_function(idx_offset, TOTAL_POINTS)

                y_idx_n = closest_idx_y-idx_offset
                y_idx_p = closest_idx_y+idx_offset
                x_idx_n = closest_idx_x-idx_offset
                x_idx_p = closest_idx_x+idx_offset

                # square sides in x direction
                for x in range(max(x_idx_n, 0), min(x_idx_p+1, nx)):
                    if y_idx_n >= 0:
                        height_map[y_idx_n, x] = (total_past_influence_map[y_idx_n, x]*height_map[y_idx_n, x] + cur_influence*cur_height) / (total_past_influence_map[y_idx_n, x] + cur_influence)
                        influence_map[y_idx_n, x] = max(cur_influence, influence_map[y_idx_n, x])
                        total_past_influence_map[y_idx_n, x] += cur_influence
                    if y_idx_p < ny:
                        height_map[y_idx_p, x] = (total_past_influence_map[y_idx_p, x]*height_map[y_idx_p, x] + cur_influence*cur_height) / (total_past_influence_map[y_idx_p, x] + cur_influence)
                        influence_map[y_idx_p, x] = max(cur_influence, influence_map[y_idx_p, x])
                        total_past_influence_map[y_idx_p, x] += cur_influence

                # square sides in y direction, excluding corners
                for y in range(max(y_idx_n+1,0), min(y_idx_p, ny)):
                    if x_idx_n >= 0:
                        height_map[y, x_idx_n] = (total_past_influence_map[y, x_idx_n]*height_map[y, x_idx_n] + cur_influence*cur_height) / (total_past_influence_map[y, x_idx_n] + cur_influence)
                        influence_map[y, x_idx_n] = max(cur_influence, influence_map[y, x_idx_n])
                        total_past_influence_map[y, x_idx_n] += cur_influence

                    if x_idx_p < nx:
                        height_map[y, x_idx_p] = (total_past_influence_map[y, x_idx_p]*height_map[y, x_idx_p] + cur_influence*cur_height) / (total_past_influence_map[y, x_idx_p] + cur_influence)
                        influence_map[y, x_idx_p] = max(cur_influence, influence_map[y, x_idx_p])
                        total_past_influence_map[y, x_idx_p] += cur_influence

    
    # temp: visualize
    # plt.matshow(influence_map, cmap='gray')
    # plt.colorbar()
    # plt.show()

    return influence_map, height_map

def trunk_height_influence_map_convex_circle(min_x, min_y, ny, nx, points_per_meter, trunk_hulls):
    # create trunk influence and height map
    influence_map = np.zeros((ny, nx), dtype= float)
    height_map = np.zeros((ny, nx), dtype=float)

    # used to keep track of influence of seen points on each points height, to do order independent weighted average
    total_past_influence_map = np.zeros((ny, nx), dtype=float)

    for tree in trunk_hulls:
        hull_3d, _ = trunk_hulls[tree]

        # for each point in hull: calculate infuence + set surrounding height
        for point in hull_3d:
            cur_height = point[2]
            
            closest_idx_x = int(np.round((point[0] - min_x ) * points_per_meter))
            closest_idx_y = int(np.round((point[1] - min_y ) * points_per_meter))
            centre_arr = np.array([closest_idx_x, closest_idx_y])

            # set influence around trunk centers in circle form
            INFLUENCE_RADIUS = 2 # radius in meters
            index_offset = INFLUENCE_RADIUS*points_per_meter
            for x in range(max(0, closest_idx_x - index_offset), min(closest_idx_x + index_offset, nx)):
                for y in range(max(0, closest_idx_y - index_offset), min(closest_idx_y + index_offset, ny)):
                    # get distance of x_indx, y_indx to closest_indx, closest_indxy
                    index_distance = np.linalg.norm(np.array([x, y]) - centre_arr)
                    actual_distance = index_distance/points_per_meter
                    
                    influence_factor = influence_function_dist(actual_distance, INFLUENCE_RADIUS)
                    if influence_factor == 0:
                        continue
                    
                    # height = weighted average between old height and current height, based on influence1
                    height_map[y, x] = (total_past_influence_map[y, x]*height_map[y, x] + influence_factor*cur_height) / (total_past_influence_map[y, x] + influence_factor)
                    # influence = max of possible influences
                    influence_map[y, x] = max(influence_factor, influence_map[y, x])
                    total_past_influence_map[y, x] += influence_factor
    
    # temp: visualize
    plt.matshow(influence_map, cmap='gray')
    plt.colorbar()
    plt.show()

    return influence_map, height_map



def points_in_hull(p, hull, tol=1e-12):
    return np.all(hull.equations[:,:-1] @ p.T + np.repeat(hull.equations[:,-1][None,:], len(p), axis=0).T <= tol, 0)

def remove_points_inside_hulls(points, hulls):
    for tree in hulls:
        _, hull_object = hulls[tree]

        max_bounds = np.amax(hull_object.points, axis=0)
        min_bounds = np.amin(hull_object.points, axis=0)

        # mask to select points within axis aligned bbox of convex hull, test these points
        bbox_mask = np.all((points[:,0:2] >= min_bounds), axis=1) & np.all((points[:,0:2] <= max_bounds), axis=1)
        
        points_in_bbox = points[bbox_mask]
        indices_in_bbox = bbox_mask.nonzero()[0]

        # in hull mask is true at indices we want to delete
        in_hull_mask = points_in_hull(points_in_bbox[:,:2], hull_object)
        indices_to_delete = indices_in_bbox[in_hull_mask]

        # deletion_mask is False at indices we want to delete
        deletion_mask = np.ones(len(points), dtype=bool)
        deletion_mask[indices_to_delete] = False
        points = points[deletion_mask]

    return points


# TODO: add understory
# plan: 
# 1. select at random from library of cut out plants
# 2. select random location, get height from terrain map (location at least 1m away from trunk locations)
# 3. get bbox of plant, check if collision with any trees
# 4. transform pc of plant until no more collision

# TODO: woody debris
# plan: 2 options
# 1. select at random from woody debris library
# 2. do similar to plants, add afterwards
# OR add in somewhere in tree placement stage, will make it likely closer to trees and better placed but complicates terrain generation etc.

# TODO: labeling of ply: probs need to convert everything to Tensor version of o3d

# TODO: postprocessing:
# downsampling
# transform everything to fit in unit cube? have seen this done in some ML papers. could also save both and compare


def terrain2mesh(terrain_cloud, decimation_factor = 5):
    # o3d triangulation does not work properly for this type of terrain mesh
    # terrain_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(terrain_cloud, alpha=1)

    tri = Delaunay(np.asarray(terrain_cloud.points)[:,:2])

    terrain_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(terrain_cloud.points), triangles=o3d.utility.Vector3iVector(tri.simplices))

    # perform decimation
    n_triangles = len(terrain_mesh.triangles)
    decimated_mesh = terrain_mesh.simplify_quadric_decimation(target_number_of_triangles=int(n_triangles//decimation_factor))

    # visualization for debug
    # terrain_mesh.translate([40,0,0])
    # decimated_mesh.translate([80,0,0])
    # o3d.visualization.draw_geometries([terrain_cloud, terrain_mesh, decimated_mesh])

    return decimated_mesh





def generate_tile(mesh_dir, pc_dir, out_dir, alpha=None):
    trees = read_trees(mesh_dir, pc_dir, alpha=alpha)

    print(f"Read {len(trees)} trees")

    plot, transforms = assemble_trees_grid(trees, n_trees=4, debug=True)
    
    # apply derived transforms to pointcloud and get trunk locations
    merged_plot = None
    trunk_hulls = {}
    for name in transforms:
        # apply transforms to open3d and merge

        pc, _, _ = trees[name]
        transform = transforms[name]
        # transform_tensor = o3d.core.Tensor(transform)
        pc = pc.transform(transform)

        hull_3d_points, hull_obj = get_trunk_convex_hull(pc)
        trunk_hulls[name] = hull_3d_points, hull_obj

        # ugly code but for some reason o3d errors when appending to empty pointcloud
        if merged_plot is None:
            # copy constructor
            merged_plot = o3d.geometry.PointCloud(pc)
        else:
            merged_plot += pc
        continue


    # add noisy terrain
    terrain_cloud = add_terrain(merged_plot, trunk_hulls)
    # merged_plot += terrain_cloud

    # get mesh of terrain
    terrain_mesh = terrain2mesh(terrain_cloud)

    
    # temp for debug
    # terrain_cloud.paint_uniform_color([1,0,0])
    # merged_plot.paint_uniform_color([0,0,1])

    # plot.show()

    o3d.visualization.draw_geometries([merged_plot, terrain_cloud])

    # TODO: write tiles here? manual check?


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