import argparse
import os
import glob
import random
import math as m
import time

import numpy as np
import open3d as o3d
import trimesh
import alphashape
import shapely
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from perlin_numpy import generate_fractal_noise_2d
from scipy import interpolate


DEBUG = False
DEBUG_ALPHA_SHAPES = False

SEMANTIC_MAP = {
    0: "terrain",
    1: "tree",
    2: "lying woody debris",
    3: "standing woody debris",
    4: "understory",
    5: "tripod" # Tom's suggestion
}

POINTS_PER_METER = 10 # NOTE: if changing this, will probably need to recalibrate a lot of the other parameters too
GRID_SIZE = 1
STEP_SIZE = 0.01 # for binning in lowest point extraction and overlaying
TREES_PER_PLOT = 9
# max size of plot
MAX_SIZE_X = 60
MAX_SIZE_Y = 60


def Tensor2VecPC(pointcloud):
    if not isinstance(pointcloud, o3d.t.geometry.PointCloud):
        print(f"Not the right type of pointcloud: {pointcloud}")
        return None
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud.point.positions.numpy()))


def read_trees(mesh_dir, pc_dir, alpha=None):

    trees = {}
    for file in sorted(glob.glob(os.path.join(pc_dir, "*.ply"))):
        pc = o3d.t.io.read_point_cloud(file)
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
        o3d_mesh = o3d.t.io.read_triangle_mesh(mesh_file)
        tri_mesh = trimesh.load(mesh_file)
        trees[name] = (pc, o3d_mesh, tri_mesh)
    return trees

def read_terrain_tiles(tile_dir):
    terrain_tiles_cuttable = []
    terrain_tiles_non_cuttable = []
    for file in sorted(glob.glob(os.path.join(tile_dir, "cuttable", "*.ply"))):
        pc = o3d.io.read_point_cloud(file)
        terrain_tiles_cuttable.append(pc)
        terrain_tiles_non_cuttable.append(pc) # all cuttable tiles can also be used in non_cuttable situations
    for file in sorted(glob.glob(os.path.join(tile_dir, "non_cuttable", "*.ply"))):
        pc = o3d.io.read_point_cloud(file)
        terrain_tiles_non_cuttable.append(pc)
    
    return [terrain_tiles_cuttable, terrain_tiles_non_cuttable]


def get_initial_tree_position(tree_mesh, pointcloud, noise_2d, max_x_row, max_y_plot):
    # start by placing tree at edge of plot

    min_x_tree, min_y_tree, min_z_tree = tree_mesh.bounds[0]

    initial_translation = np.array([max_x_row-min_x_tree, max_y_plot-min_y_tree, -min_z_tree])
    bbox_transform = trimesh.transformations.translation_matrix(initial_translation)

    inverse_bbox_transform = trimesh.transformations.translation_matrix(-initial_translation)

    pointcloud = pointcloud.transform(bbox_transform)

    # set height based on local noise value
    trunk_center_x, trunk_center_y, _ = get_trunk_location(pointcloud)

    # do the inverse transform after getting the correct location, because transform is in place
    # this seems a little weird, but in our workflow the transforms are all saved
    # NOTE: possible to change up workflow and apply transforms directly, but this works fine
    pointcloud = pointcloud.transform(inverse_bbox_transform)

    noise_idx_x = round(POINTS_PER_METER * trunk_center_x)
    noise_idx_y = round(POINTS_PER_METER * trunk_center_y)
    noise_idx_x = min(len(noise_2d)-1, noise_idx_x)
    noise_idx_y = min(len(noise_2d[0])-1, noise_idx_y)

    z_target = noise_2d[noise_idx_x][noise_idx_y]


    height_translation = np.array([0, 0, z_target])

    total_transform = trimesh.transformations.translation_matrix(initial_translation + height_translation)

    tree_mesh.apply_transform(total_transform)

    return tree_mesh, total_transform

def move_tree_closer(name, tree_mesh, collision_manager_plot, collision_manager_row, trees, max_x_row, debug=False):

    # move tree as close as possible to rest of plot
    total_translation = np.array([0.0,0.0,0.0])
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

def assemble_trees_grid(trees, terrain_noise, n_trees=9, debug=False):
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
        
        # TODO: TEMP for quick tests
        # if TREES_PER_PLOT == 1:
        #     name = 'wytham_winter_5b'

        pc, _, tri_mesh = trees[name]

        # save all transforms for this tree to single matrix
        o3d_transform = np.identity(4, dtype=float)

        if not debug:
            # generate random rotation around z axis
            rot_angle = m.radians(random.randrange(360))
            rot_matrix = np.identity(4, dtype=float)
            rot_matrix[:2,:2] = [[m.cos(rot_angle), -m.sin(rot_angle)], [m.sin(rot_angle),m.cos(rot_angle)]]

            # save rotation and apply to trimesh mesh
            # o3d_transform = np.matmul(rot_matrix, o3d_transform)
            pc = pc.transform(rot_matrix)
            tri_mesh.apply_transform(rot_matrix)


        if i == 0:
            # start plot at 0,0,0
            print(f"Placing tree {name}")

            tri_mesh, origin_translation = get_initial_tree_position(tri_mesh, pc, terrain_noise, 0, 0)

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
            
            tree_mesh, initial_translation = get_initial_tree_position(tri_mesh, pc, terrain_noise, max_x_row, max_y_plot)

            tree_mesh, second_translation, max_x_row = move_tree_closer(name, tree_mesh, collision_manager_plot, collision_manager_row, trees, max_x_row, debug=debug)

            # tree_mesh, translation, max_x_row = place_tree_in_grid(name, tri_mesh, collision_manager_plot, collision_manager_row, trees, max_x_row, max_y_plot, debug=debug)
            collision_meshes[name] = tree_mesh
            tri_mesh_plot += tree_mesh
            if debug:
                tri_mesh_plot += tree_mesh.bounding_box
            o3d_transform = np.matmul(initial_translation, o3d_transform)
            o3d_transform = np.matmul(second_translation, o3d_transform)

        # save_transforms
        transforms[name] = o3d_transform
    
    return tri_mesh_plot, transforms



def get_trunk_location(pointcloud):
    # get axis aligned bounding box
    aaligned_bbox = pointcloud.get_axis_aligned_bounding_box()

    # slice bottom meter of bounding box
    min_bound_bbox = aaligned_bbox.min_bound
    max_bound_bbox = aaligned_bbox.max_bound
    max_bound_bbox[2] = min_bound_bbox[2] + 1
    aaligned_bbox.set_max_bound(max_bound_bbox)

    # crop pointcloud to bottom bbox
    cropped_pointcloud = pointcloud.crop(aaligned_bbox)

    cropped_bbox = cropped_pointcloud.get_axis_aligned_bounding_box()

    center_x, center_y = cropped_bbox.get_center().numpy()[:2]
    # Temporary: get convex hull center
    return [center_x, center_y, min_bound_bbox[2]]

def get_trunk_convex_hull(pointcloud, slice_height=0.75):  
    # get axis aligned bounding box
    aaligned_bbox = pointcloud.get_axis_aligned_bounding_box()

    # slice bottom part of bounding box
    min_bound_bbox = aaligned_bbox.min_bound
    max_bound_bbox = aaligned_bbox.max_bound
    max_bound_bbox[2] = min_bound_bbox[2] + slice_height
    aaligned_bbox.set_max_bound(max_bound_bbox)

    # crop pointcloud to bottom bbox
    cropped_pointcloud = pointcloud.crop(aaligned_bbox)

    points_3d = cropped_pointcloud.point.positions.numpy()
    points_projected_2d = points_3d[:,:2]

    hull = ConvexHull(points_projected_2d)
    
    hull_points_3d = points_3d[hull.vertices]
    return hull_points_3d, hull

def get_trunk_alpha_shape(pointcloud, name, slice_height=0.75):
    debug = DEBUG_ALPHA_SHAPES

    # get axis aligned bounding box
    aaligned_bbox = pointcloud.get_axis_aligned_bounding_box()

    # slice bottom part of bounding box
    min_bound_bbox = aaligned_bbox.min_bound
    max_bound_bbox = aaligned_bbox.max_bound
    max_bound_bbox[2] = min_bound_bbox[2] + slice_height
    aaligned_bbox.set_max_bound(max_bound_bbox)

    # crop pointcloud to bottom bbox
    cropped_pointcloud = pointcloud.crop(aaligned_bbox)

    points_3d = cropped_pointcloud.point.positions.numpy()
    points_projected_2d = points_3d[:,:2]

    ALPHA = 10.0
    alpha_shape = alphashape.alphashape(points_projected_2d, ALPHA)
    second_try = False

    # check if alphashape area is not inside tree by checking that area is large enough + check if not split up
    # if so, decrease alpha
    bbox_area = np.prod(np.amax(points_projected_2d, axis=0) - np.amin(points_projected_2d, axis=0))
    while isinstance(alpha_shape, shapely.MultiPolygon) or alpha_shape.area < 0.4*bbox_area:
        ALPHA -= 1
        if ALPHA <= 0:
            if not second_try:
                print(f"Problem: got to alpha is 0, alphashape package hangs in this case, for pc {name}")
                
                if debug:
                    fig, ax = plt.subplots()
                    ax.scatter(*zip(*points_projected_2d))
                    if not isinstance(alpha_shape, shapely.MultiPolygon):
                        x,y = alpha_shape.exterior.xy
                        plt.plot(x,y)
                    else:
                        for geom in alpha_shape.geoms:
                            plt.plot(*geom.exterior.xy)
                    plt.show()
                ALPHA = 10.0
                max_bound_bbox[2] = min_bound_bbox[2] + slice_height*2
                aaligned_bbox.set_max_bound(max_bound_bbox)
                cropped_pointcloud = pointcloud.crop(aaligned_bbox)
                points_3d = cropped_pointcloud.point.positions.numpy()
                points_projected_2d = points_3d[:,:2]
                second_try = True
            else:
                print(f"Bigger problem: got to alpha is 0 on second try for pc {name}")
                print("Just accepting what we have now")
                break


        alpha_shape = alphashape.alphashape(points_projected_2d, ALPHA)

    # temp: plot alpha shape
    if second_try and debug:
        print("Plotting after second try")
        fig, ax = plt.subplots()
        ax.scatter(*zip(*points_projected_2d))
        if not isinstance(alpha_shape, shapely.MultiPolygon):
            x,y = alpha_shape.exterior.xy
            plt.plot(x,y)
        else:
            for geom in alpha_shape.geoms:
                plt.plot(*geom.exterior.xy)
        plt.show()

    return alpha_shape



def perlin_terrain():
    nx = round(POINTS_PER_METER * MAX_SIZE_X) + 1
    ny = round(POINTS_PER_METER * MAX_SIZE_Y) + 1

    # perlin noise settings
    RES = 3
    LACUNARITY = 2
    OCTAVES = 6
    PERSISTENCE = 0.45
    SCALE = random.uniform(2.5, 4)

    # get dimensions to generate perlin noise, shape must be multiple of res*lacunarity**(octaves - 1)
    shape_factor = RES*(LACUNARITY**(OCTAVES-1))
    perlin_nx = nx - (nx % shape_factor) + shape_factor
    perlin_ny = ny - (ny % shape_factor) + shape_factor

    perlin_noise = generate_fractal_noise_2d((perlin_nx, perlin_ny), (RES, RES), octaves=OCTAVES, lacunarity=LACUNARITY, persistence=PERSISTENCE)
    perlin_noise = perlin_noise[:nx, :ny]

    perlin_noise = perlin_noise * SCALE
    
    return perlin_noise



def influence_function_dist(distance, max_distance):
    if distance >= max_distance:
        return 0
    x = distance/max_distance
    return (x-1)**2

def trunk_height_influence_map_convex_circle(min_x, min_y, nx, ny, points_per_meter, trunk_hulls):
    # create trunk influence and height map
    influence_map = np.zeros((nx, ny), dtype= float)
    height_map = np.zeros((nx, ny), dtype=float)

    # used to keep track of influence of seen points on each points height, to do order independent weighted average
    total_past_influence_map = np.zeros((nx, ny), dtype=float)

    for tree in trunk_hulls:
        hull_3d, _ = trunk_hulls[tree]

        # for each point in hull: calculate infuence + set surrounding height
        for point in hull_3d:
            cur_height = point[2]
            
            closest_idx_x = int(np.round((point[0] - min_x ) * points_per_meter))
            closest_idx_y = int(np.round((point[1] - min_y ) * points_per_meter))
            centre_arr = np.array([closest_idx_x, closest_idx_y])

            # set influence around trunk centers in circle form
            INFLUENCE_RADIUS = 3 # radius in meters
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
                    height_map[x, y] = (total_past_influence_map[x, y]*height_map[x, y] + influence_factor*cur_height) / (total_past_influence_map[x, y] + influence_factor)
                    # influence = max of possible influences
                    influence_map[x, y] = max(influence_factor, influence_map[x, y])
                    total_past_influence_map[x, y] += influence_factor
    
    # temp: visualize
    # plt.matshow(influence_map, cmap='gray')
    # plt.colorbar()
    # plt.show()

    return influence_map, height_map



def extract_lowest(terrain_tile, step_size=STEP_SIZE):
    # divide tile into bins of step_size and select bottom points in each bin by comparing to neighbouring bins

    min_bound = terrain_tile.get_min_bound()
    max_bound = terrain_tile.get_max_bound()

    x_range = np.arange(min_bound[0], max_bound[0], step=step_size)
    y_range = np.arange(min_bound[1], max_bound[1], step=step_size)

    # bottom_cloud = o3d.geometry.PointCloud()
    # top_cloud = o3d.geometry.PointCloud()

    x_bins = len(x_range)
    y_bins = len(y_range)

    lowest = np.ones((x_bins, y_bins, 3))*np.inf
    points_arr = np.asarray(terrain_tile.points)

    bins = [[[] for y in range(y_bins)] for x in range(x_bins)]

    for point in points_arr:
        bin_idx = (point - min_bound) // step_size
        idx_x = int(bin_idx[0])
        idx_y = int(bin_idx[1])
        bins[idx_x][idx_y].append(point)
        if point[2] < lowest[idx_x][idx_y][2]:
            lowest[idx_x][idx_y] = point
        
    points_bot = []
    points_top = []
    bins_lowest = [[None for y in range(y_bins)] for x in range(x_bins)]
    for point in points_arr:
        bin_idx = (point - min_bound) // step_size
        i = int(bin_idx[0])
        j = int(bin_idx[1])
        i_min = max(0, i-1)
        j_min = max(0, j-1)
        i_max = min(i+1, x_bins-1)
        j_max = min(j+1, y_bins-1)
        remove = False
        for x in range(i_min, i_max+1):
            for y in range(j_min, j_max+1):
                diff = point - lowest[x][y]
                dist2d = diff[0]*diff[0] + diff[1]*diff[1]
                if diff[2] > 0 and diff[2]*diff[2] > dist2d:
                    remove = True
                    break
            if remove:
                break
        if remove:
            points_top.append(point)
        else:
            points_bot.append(point)
            bins_lowest[i][j] = point
    
    # bottom_cloud = o3d.geometry.PointCloud()
    # bottom_cloud.points = o3d.utility.Vector3dVector(np.array(points_bot))
    # bottom_cloud.paint_uniform_color(np.array([0,1,0]))
    # top_cloud = o3d.geometry.PointCloud()
    # top_cloud.points = o3d.utility.Vector3dVector(np.array(points_top))
    # top_cloud.paint_uniform_color(np.array([1,0,0]))
    return bins, bins_lowest

def precalc_lowest(terrain_tiles):
    bin_arr = []
    for tile in terrain_tiles:
        bins_tuple = extract_lowest(tile)
        bin_arr.append(bins_tuple)

    return bin_arr

def trunk_in_tile(cur_noise_tile, trunk_corner_points):
    noise_max = np.amax(cur_noise_tile, axis=0)[:2]
    noise_min = np.amin(cur_noise_tile, axis=0)[:2]

    # check if any corner point inside tile
    point_in_tile = np.any(np.all(np.logical_and(noise_min <= trunk_corner_points, trunk_corner_points <= noise_max), axis=1)) # expl: logical and for upper and lower bound, np.all along axis 1 to check both x and y, then np.any to check all points
    return point_in_tile

def get_closest_lowest(bins_lowest, i, j, point):
    # if no lowest point at i,j, look in tiles around it for a lowest point by checking moore neighboorhoud for closest lowest point
    x_bins = len(bins_lowest)
    y_bins = len(bins_lowest[0])

    index_offset = 1
    found = False
    while not found:
        i_min = max(0, i-index_offset)
        j_min = max(0, j-index_offset)
        i_max = min(i+index_offset, x_bins-1)
        j_max = min(j+index_offset, y_bins-1)
        closest_dist = np.inf
        closest_xy = None
        for x in (i_min, i_max):
            for y in (j_min, j_max):
                if bins_lowest[x][y] is not None:
                    dist2d = np.linalg.norm(point - bins_lowest[x][y][:2])
                    found = True
                    if dist2d < closest_dist:
                        closest_dist = dist2d
                        closest_xy = (x,y)
        if found:
            return bins_lowest[closest_xy[0]][closest_xy[1]]
        index_offset += 1

def overlay_single_tile(terrain_tile, noise_tile, interpolator, bins_tuple):
    # bins, bins_lowest = extract_lowest_alt(terrain_tile, step_size=STEP_SIZE)
    bins, bins_lowest = bins_tuple

    all_points = []

    # move terrain tile to noise 
    min_bounds_noise = np.min(noise_tile, axis=0)
    terrain_tile = terrain_tile.translate(min_bounds_noise-terrain_tile.get_min_bound())

    max_bounds_noise = np.max(noise_tile, axis=0)
    extent_noise = max_bounds_noise - min_bounds_noise

    max_num_bins_x = m.floor((1/STEP_SIZE)*extent_noise[0])
    max_num_bins_y = m.floor((1/STEP_SIZE)*extent_noise[1])

    # for all bins within noise tile
    for i in range(max_num_bins_x):
        for j in range(max_num_bins_y):
            if len(bins[i][j]) == 0:
                continue

            bin_center_x = i*STEP_SIZE+STEP_SIZE/2 + min_bounds_noise[0]
            bin_center_y = j*STEP_SIZE+STEP_SIZE/2 + min_bounds_noise[1]

            center_arr = np.array([bin_center_x, bin_center_y])

            # get closest lowest point
            if bins_lowest[i][j] is None:
                lowest_point = np.array(get_closest_lowest(bins_lowest, i, j, center_arr))
            else:
                lowest_point = bins_lowest[i][j]

            # correct height of all points in tile to interpolated perlin noise height
            # we estimate the height of the terrain tile as the height of the closest lowest point
            height = interpolator((bin_center_x, bin_center_y))
            height_correction = height - lowest_point[2]

            points = np.array(bins[i][j])
            corrected_points = points + np.array([0, 0, height_correction])
            all_points.append(corrected_points)
            
    corrected_cloud = o3d.geometry.PointCloud()
    corrected_cloud.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    # corrected_cloud.paint_uniform_color(np.array([0,1,0]))

    noise_cloud = o3d.geometry.PointCloud()
    noise_cloud.points = o3d.utility.Vector3dVector(np.array(noise_tile))
    noise_cloud.paint_uniform_color(np.array([1,0,0]))

    # o3d.visualization.draw_geometries([noise_cloud, corrected_cloud])

    return noise_cloud, corrected_cloud

def overlay_terrain(noise_2D, noise_coordinates, interpolator, terrain_tiles, trunk_hulls):

    # get trunk bbox corner points
    trunk_corner_points = []
    for trunk in trunk_hulls:
        trunk_3d, _ = trunk_hulls[trunk]
        hull_points_2d = trunk_3d[:,:2]
        max_hull_2d = np.amax(hull_points_2d, axis=0)
        min_hull_2d = np.amin(hull_points_2d, axis=0)
        trunk_corner_points.append([min_hull_2d[0], min_hull_2d[1]])
        trunk_corner_points.append([min_hull_2d[0], max_hull_2d[1]])
        trunk_corner_points.append([max_hull_2d[0], min_hull_2d[1]])
        trunk_corner_points.append([max_hull_2d[0], max_hull_2d[1]])
    
    trunk_corner_points = np.array(trunk_corner_points)

    # get dimensions of noise terrain to fill up
    pptile = GRID_SIZE*POINTS_PER_METER
    n_x = len(noise_2D)
    n_y = len(noise_2D[1])
    n_tiles_x = m.ceil((n_x-1) / pptile)
    x_edge_points_rng = (n_x-1) % pptile
    n_tiles_y = m.ceil((n_y-1) / pptile)
    y_edge_points_rng = (n_x-1) % pptile

    # prepare cuttable and non cuttable tiles
    cuttable_tiles = terrain_tiles[0]
    non_cuttable_tiles = terrain_tiles[1]
    random.shuffle(cuttable_tiles)
    random.shuffle(non_cuttable_tiles)
    cuttable_idx = 0
    non_cuttable_idx = 0
    n_cuttable = len(cuttable_tiles)
    n_non_cuttable = len(non_cuttable_tiles)

    # pre extract binned tile and lowest points
    precalc = False
    if (n_tiles_x*n_tiles_y) > (n_cuttable + n_non_cuttable):
        print("Precalculating binned lowest points")
        bins_cuttable = precalc_lowest(cuttable_tiles)
        bins_non_cuttable = precalc_lowest(non_cuttable_tiles)
        precalc = True

    
    print(f"Overlaying noise, {n_tiles_x*n_tiles_y} tiles of size {GRID_SIZE}x{GRID_SIZE}")
    print(f"Selecting from library of {n_cuttable} cuttable tiles and {n_non_cuttable} noncuttable tiles")

    merged_terrain_cloud = None

    tiles = []
    for i in range(n_tiles_x):
        for j in range(n_tiles_y):
            # slice coordinate list
            cur_noise_tile = []

            # need to get 2D tile from list of points, do some indexing magic based on the meshgrid format
            x_index_range = pptile + 1
            y_index_range = pptile + 1
            # handle edge cases, little ugly but oh well
            if i == (n_tiles_x-1) and x_edge_points_rng != 0:
                x_index_range = x_edge_points_rng + 1
            if j == (n_tiles_y - 1) and y_edge_points_rng != 0:
                y_index_range = y_edge_points_rng + 1
            
            # shift in array of points, x is row index, y is column index
            shift = i*pptile+j*n_x*pptile
            for l in range(y_index_range):
                cur_noise_tile.append(noise_coordinates[shift+l*n_x:shift+l*n_x+x_index_range])
            cur_noise_tile = np.vstack(cur_noise_tile)

            # select tile
            if trunk_in_tile(cur_noise_tile, trunk_corner_points):
                cur_terrain_tile = cuttable_tiles[cuttable_idx]
                if precalc:
                    bins_tuple = bins_cuttable[cuttable_idx]
                else:
                    bins_tuple = extract_lowest(cur_terrain_tile)
                cuttable_idx += 1
                cuttable_idx = cuttable_idx % n_cuttable
                if cuttable_idx == 0:
                    if precalc:
                        # reshuffle tiles and bins in same order
                        c = list(zip(cuttable_tiles, bins_cuttable))
                        random.shuffle(c)
                        cuttable_tiles, bins_cuttable = zip(*c)
                    else:
                        random.shuffle(cuttable_tiles)
            else:
                cur_terrain_tile = non_cuttable_tiles[non_cuttable_idx]
                if precalc:
                    bins_tuple = bins_non_cuttable[non_cuttable_idx]
                else:
                    bins_tuple = extract_lowest(cur_terrain_tile)
                non_cuttable_idx += 1
                non_cuttable_idx = non_cuttable_idx % n_non_cuttable
                if non_cuttable_idx == 0:
                    if precalc:
                        # reshuffle tiles and bins in same order
                        c = list(zip(non_cuttable_tiles, bins_non_cuttable))
                        random.shuffle(c)
                        non_cuttable_tiles, bins_non_cuttable = zip(*c)
                    else:
                        random.shuffle(non_cuttable_tiles)
            
            # overlay single noise tile on noise
            noise_cloud, tile_cloud = overlay_single_tile(cur_terrain_tile, cur_noise_tile, interpolator, bins_tuple)
            tiles.append(tile_cloud)
            tiles.append(noise_cloud)

            if merged_terrain_cloud is None:
                # copy constructor
                merged_terrain_cloud = o3d.geometry.PointCloud(tile_cloud)
            else:
                merged_terrain_cloud += tile_cloud
    return merged_terrain_cloud
    



def points_in_alphashape(points, alphashape):
    mask = []
    for point in points:
        mask.append(alphashape.contains(shapely.Point(point)))
    return np.array(mask)

def remove_points_inside_alpha_shape(points, alphashapes):
    for tree in alphashapes:
        polygon = alphashapes[tree]

        min_bounds = polygon.bounds[:2]
        max_bounds = polygon.bounds[2:4]

        # mask to select points within axis aligned bbox of convex hull, test these points
        bbox_mask = np.all((points[:,0:2] >= min_bounds), axis=1) & np.all((points[:,0:2] <= max_bounds), axis=1)
        
        points_in_bbox = points[bbox_mask]
        indices_in_bbox = bbox_mask.nonzero()[0]

        # in hull mask is true at indices we want to delete
        in_hull_mask = points_in_alphashape(points_in_bbox[:,:2], polygon)
        if len(in_hull_mask) > 0:
            indices_to_delete = indices_in_bbox[in_hull_mask]

        # deletion_mask is False at indices we want to delete
        deletion_mask = np.ones(len(points), dtype=bool)
        deletion_mask[indices_to_delete] = False
        points = points[deletion_mask]

    return points


def build_terrain(plot_cloud, perlin_noise, trunk_hulls, alphashapes, terrain_tiles):
    # get dimension of plot cloud
    max_x, max_y, _ = plot_cloud.get_max_bound().numpy()
    min_x, min_y, _ = plot_cloud.get_min_bound().numpy()

    nx = round((max_x - min_x) * POINTS_PER_METER) + 1
    ny = round((max_y - min_y) * POINTS_PER_METER) + 1

    # cut perlin noise to size of plot
    perlin_noise = perlin_noise[:nx, :ny]

    print("Blending noise with trunk height map")
    # get influence map and height map of trunks to adapt terrain to trunk heights and locations
    influence_map, height_map = trunk_height_influence_map_convex_circle(min_x, min_y, nx, ny, POINTS_PER_METER, trunk_hulls=trunk_hulls)
    final_xy_map = influence_map * height_map + (np.ones(influence_map.shape, dtype=float) - influence_map) * perlin_noise

    # visualization
    # plt.matshow(final_xy_map, cmap='gray', interpolation='lanczos')
    # plt.colorbar()
    # plt.show()

    # get xy grid
    x = np.linspace(min_x, max_x, num = nx)
    y = np.linspace(min_y, max_y, num = ny)
    xv, yv = np.meshgrid(x, y)
    
    # apply height map
    points_xy = np.array([xv.flatten(), yv.flatten()]).T
    z_arr = final_xy_map.T.flatten()
    points_3d = np.column_stack((points_xy, z_arr))


    interpolator = interpolate.RegularGridInterpolator((x,y), final_xy_map)

    print("Overlaying real terrain tiles")
    merged_terrain_cloud = overlay_terrain(final_xy_map, points_3d, interpolator, terrain_tiles, trunk_hulls)
    points_3d_real = np.asarray(merged_terrain_cloud.points)

    print("Removing terrain inside tree trunks")
    points_3d_cleaned = remove_points_inside_alpha_shape(points_3d_real, alphashapes)

    # to pointcloud
    tensor_3d = o3d.core.Tensor(points_3d_cleaned.astype(np.float32))
    terrain_cloud = o3d.t.geometry.PointCloud()
    terrain_cloud.point.positions = tensor_3d
    # add labels to terrain cloud: semantic terrain label is 0, no instance so -1
    terrain_cloud.point.semantic = o3d.core.Tensor(np.zeros(len(points_3d_cleaned), dtype=np.int32)[:,None])
    terrain_cloud.point.instance = o3d.core.Tensor((-1)*np.ones(len(points_3d_cleaned), dtype=np.int32)[:,None])

    # TODO: TEMP: for wouter: add terrain label to "labels" field
    terrain_cloud.point.labels = o3d.core.Tensor(2*np.ones(len(points_3d_cleaned), dtype=np.int32)[:,None])

    return terrain_cloud



def save_tile(out_dir, out_pc, tile_id, downsampled=True):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"Tile_{tile_id}.ply")

    while os.path.exists(out_path):
        tile_id += 1
        out_path = os.path.join(out_dir, f"Tile_{tile_id}.ply")

    o3d.t.io.write_point_cloud(out_path, out_pc)

    # possibly also write downsampled version
    if downsampled:
        pc_downsampled = out_pc.voxel_down_sample(voxel_size=0.02)

        out_dir_ds = os.path.join(out_dir, "downsampled")
        if not os.path.exists(out_dir_ds):
            os.mkdir(out_dir_ds)

        out_path_ds = os.path.join(out_dir_ds, f"Tile_{tile_id}.ply")
        
        o3d.t.io.write_point_cloud(out_path_ds, pc_downsampled)
    return

def generate_tile(trees, terrain_tiles, debug=DEBUG):

    terrain_noise = perlin_terrain()

    # get variation in number of trees
    if not debug:
        n_trees = random.randint(max(TREES_PER_PLOT-3,0), TREES_PER_PLOT+3)
    else:
        n_trees = TREES_PER_PLOT

    plot, transforms = assemble_trees_grid(trees, terrain_noise, n_trees=n_trees, debug=debug)
    
    # apply derived transforms to pointcloud and get trunk locations
    merged_cloud = None
    if debug:
        merged_plot_debug = None

    trunk_hulls = {}
    alphashapes = {}

    cur_instance_id = 1

    cmap = plt.get_cmap("Set1")
    n_colors = len(cmap.colors)


    # apply transforms to open3d and merge
    for name in transforms:

        pc, _, _ = trees[name]
        transform = transforms[name]
        # transform_tensor = o3d.core.Tensor(transform)
        pc = pc.transform(transform)

        # add labels to cloud: semantic trees label is 1, unique instance label per tree
        pc.point.semantic = o3d.core.Tensor(np.ones(len(pc.point.positions), dtype=np.int32)[:,None])
        pc.point.instance = o3d.core.Tensor(cur_instance_id*np.ones(len(pc.point.positions), dtype=np.int32)[:,None])
        cur_instance_id += 1

        # get hull and alpha_shape of pc in merged cloud
        trunk_hulls[name] = get_trunk_convex_hull(pc)
        alphashapes[name] = get_trunk_alpha_shape(pc, name)

        # append point cloud
        if merged_cloud is None:
            if debug:
                merged_plot_debug = Tensor2VecPC(pc).paint_uniform_color(cmap.colors[(cur_instance_id-2) % n_colors])
            # copy constructor
            merged_cloud = o3d.t.geometry.PointCloud(pc)
        else:
            merged_cloud += pc
            if debug:
                merged_plot_debug += Tensor2VecPC(pc).paint_uniform_color(cmap.colors[(cur_instance_id-2) % n_colors])
        

    x_size, y_size, _ = merged_cloud.get_max_bound().numpy() - merged_cloud.get_min_bound().numpy()

    if x_size > MAX_SIZE_X or y_size > MAX_SIZE_Y:
        print(f"Generated plot is too big ({x_size}, {y_size}), writing downsampled version to seperate folder and returning")
        merged_cloud = merged_cloud.voxel_down_sample(voxel_size=0.05)
        return merged_cloud, False

    if debug:
        plot.show()
        o3d.visualization.draw_geometries([merged_plot_debug])

    # after placing all trees, merge perlin terrain with trees using hulls
    terrain_cloud = build_terrain(merged_cloud, terrain_noise, trunk_hulls, alphashapes, terrain_tiles)

    # TODO: TEMP for vis
    # o3d.t.io.write_point_cloud("assets/terrain_merged_isolated.ply", terrain_cloud)

    # downsample terrain to limit mem
    terrain_cloud_ds = terrain_cloud.voxel_down_sample(voxel_size=0.025)

    # o3d.t.io.write_point_cloud("assets/trees_isolated.ply", merged_cloud)


    if debug:
        terrain_cloud_debug = Tensor2VecPC(terrain_cloud)
        terrain_cloud_debug.paint_uniform_color([0.75,0.75,0.75])
    
    if debug:
        o3d.visualization.draw_geometries([merged_plot_debug, terrain_cloud_debug])

    merged_cloud += terrain_cloud_ds

    return merged_cloud, True

def generate_tiles(mesh_dir, pc_dir, tiles_dir, out_dir, alpha=None, n_tiles=10):
    # read trees
    trees = read_trees(mesh_dir, pc_dir, alpha=alpha)

    terrain_tiles = read_terrain_tiles(tiles_dir)

    print(f"Read {len(trees)} trees")

    print(f"Generating {n_tiles} tiles")
    tile_id = 0
    while tile_id < n_tiles:

        start_time = time.process_time()
        tile_cloud, tile_ok = generate_tile(trees, terrain_tiles)
        end_time = time.process_time()
        print(f"Generated tile {tile_id+1} in {end_time-start_time} seconds")
        if tile_ok:
            save_tile(out_dir, tile_cloud, tile_id)
            tile_id += 1
        else:
            save_tile(os.path.join(out_dir, "too_large_debug"), tile_cloud, tile_id, downsampled=False)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pointcloud_directory", required=True)
    parser.add_argument("-t", "--tile_directory", required=True)
    parser.add_argument("-d", "--mesh_directory", default=None)
    parser.add_argument("-o", "--output_directory", default=None)
    parser.add_argument("-n", "--n_tiles", default=10, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.pointcloud_directory):
        print(f"Couldn't read trees input dir {args.pointcloud_directory}!")
        return
    
    if not os.path.exists(os.path.join(args.tile_directory, "cuttable")) or not os.path.exists(os.path.join(args.tile_directory, "non_cuttable")):
        print(f"Couldn't read cuttable or non_cuttable tile input dirs in {args.tile_directory}!")
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
        out_dir = os.path.join(args.pointcloud_directory, "synthetic_tiles")
    
    generate_tiles(args.mesh_directory, args.pointcloud_directory, args.tile_directory, out_dir, n_tiles=args.n_tiles)

    # trees = read_trees(args.mesh_directory, args.pointcloud_directory)
    # terrain_tiles = read_terrain_tiles(args.tile_directory)

    # tile_cloud, _ = generate_tile(trees, terrain_tiles)

    # print(tile_cloud.get_max_bound() - tile_cloud.get_min_bound())

    # tile_vec = Tensor2VecPC(tile_cloud)

    # o3d.visualization.draw_geometries([tile_vec])

    return



if __name__ == "__main__":
    main()