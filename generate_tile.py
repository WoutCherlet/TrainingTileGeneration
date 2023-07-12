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

from preprocess_terrain import overlay_terrain


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
TREES_PER_PLOT = 1 # NOTE: slight variations in this to get different amount of instances per plot?
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
    terrain_tiles = []
    for file in sorted(glob.glob(os.path.join(tile_dir, "*.ply"))):
        pc = o3d.io.read_point_cloud(file)
        terrain_tiles.append(pc)
    
    return terrain_tiles


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
    merged_terrain_cloud = overlay_terrain(final_xy_map, points_3d, interpolator, terrain_tiles)
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

    return terrain_cloud



def save_tile(out_dir, out_pc, tile_id, downsampled=True):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

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

    plot, transforms = assemble_trees_grid(trees, terrain_noise, n_trees=TREES_PER_PLOT, debug=debug)
    
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


    if debug:
        terrain_cloud_debug = Tensor2VecPC(terrain_cloud)
        terrain_cloud_debug.paint_uniform_color([0.75,0.75,0.75])
    
    if debug:
        o3d.visualization.draw_geometries([merged_plot_debug, terrain_cloud_debug])

    merged_cloud += terrain_cloud

    return merged_cloud, True

def generate_tiles(mesh_dir, pc_dir, tiles_dir, out_dir, alpha=None, n_tiles=10):
    # read trees
    trees = read_trees(mesh_dir, pc_dir, alpha=alpha)

    # TODO: expand to cuttable and non_cuttable
    terrain_tiles = read_terrain_tiles(tiles_dir)

    print(f"Read {len(trees)} trees")

    print(f"Generating {n_tiles}")
    for i in range(n_tiles):

        start_time = time.process_time()
        tile_cloud, tile_ok = generate_tile(trees, terrain_tiles)
        end_time = time.process_time()
        print(f"Generated tile {i+1} in {end_time-start_time} seconds")
        if tile_ok:
            save_tile(out_dir, tile_cloud, i+1)
        else:
            save_tile(os.path.join(out_dir, "too_large_debug"), tile_cloud, i+1, downsampled=False)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pointcloud_directory", required=True)
    parser.add_argument("-t", "--tile_directory", required=True)
    parser.add_argument("-d", "--mesh_directory", default=None)
    parser.add_argument("-o", "--output_directory", default=None)
    parser.add_argument("-n", "--n_tiles", default=10)

    args = parser.parse_args()

    if not os.path.exists(args.pointcloud_directory):
        print(f"Couldn't read trees input dir {args.pointcloud_directory}!")
        return
    
    if not os.path.exists(args.tile_directory):
        print(f"Couldn't read tile input dir {args.tile_directory}!")
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
    
    # generate_tiles(args.mesh_directory, args.pointcloud_directory, args.tile_directory, out_dir, n_tiles=args.n_tiles)

    trees = read_trees(args.mesh_directory, args.pointcloud_directory)
    terrain_tiles = read_terrain_tiles(args.tile_directory)

    tile_cloud, _ = generate_tile(trees, terrain_tiles)

    print(tile_cloud.get_max_bound() - tile_cloud.get_min_bound())

    tile_vec = Tensor2VecPC(tile_cloud)

    o3d.visualization.draw_geometries([tile_vec])

    return



if __name__ == "__main__":
    main()