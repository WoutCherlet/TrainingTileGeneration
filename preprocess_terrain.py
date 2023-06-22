import argparse
import os
import math as m
import open3d as o3d
import numpy as np
from perlin_numpy import generate_fractal_noise_2d
from scipy import interpolate


POINTS_PER_METER = 10
GRID_SIZE = 1 # in metres

def tile_is_square_of_gridsize(tile_cloud, GRID_SIZE):
    BOUND = GRID_SIZE*0.1

    tile_min_bound = tile_cloud.get_min_bound()
    tile_max_bound = tile_cloud.get_max_bound()

    # check if any points in all corners
    bl_min_bound = tile_min_bound
    bl_max_bound = tile_min_bound + np.array([BOUND, BOUND, tile_max_bound[2]])
    bl_corner = o3d.geometry.AxisAlignedBoundingBox(min_bound=bl_min_bound, max_bound=bl_max_bound)
    bl_corner_cloud = tile_cloud.crop(bl_corner)
    tl_min_bound = tile_min_bound + np.array([0, GRID_SIZE-BOUND, 0])
    tl_max_bound = tile_min_bound + np.array([BOUND, GRID_SIZE, tile_max_bound[2]])
    tl_corner = o3d.geometry.AxisAlignedBoundingBox(min_bound=tl_min_bound, max_bound=tl_max_bound)
    tl_corner_cloud = tile_cloud.crop(tl_corner)
    br_min_bound = tile_min_bound + np.array([GRID_SIZE - BOUND, 0, 0])
    br_max_bound = tile_min_bound + np.array([GRID_SIZE, BOUND, tile_max_bound[2]])
    br_corner = o3d.geometry.AxisAlignedBoundingBox(min_bound=br_min_bound, max_bound = br_max_bound)
    br_corner_cloud = tile_cloud.crop(br_corner)
    tr_min_bound = tile_min_bound + np.array([GRID_SIZE - BOUND, GRID_SIZE - BOUND, 0])
    tr_max_bound = tile_min_bound + np.array([GRID_SIZE, GRID_SIZE, tile_max_bound[2]])
    tr_corner = o3d.geometry.AxisAlignedBoundingBox(min_bound=tr_min_bound, max_bound = tr_max_bound)
    tr_corner_cloud = tile_cloud.crop(tr_corner)

    corners_empty = tl_corner_cloud.is_empty() or br_corner_cloud.is_empty() or tr_corner_cloud.is_empty() or bl_corner_cloud.is_empty()

    return not corners_empty

def preprocess_terrain(terrain_cloud):
    pc = o3d.io.read_point_cloud(terrain_cloud)

    # TODO: we can get more tiles out of same plot if we rotate longest side to be parallel to xy (might not be worth the effort)

    min_bound = pc.get_min_bound()
    max_bound = pc.get_max_bound()
    pc_height = max_bound[2] - min_bound[2]

    bins_x, bins_y, _ = (max_bound - min_bound) // GRID_SIZE
    tiles = []

    visualization_shift = 0.10

    for i in range(0, int(bins_x)):
        for j in range(0, int(bins_y)):
            crop_min_bound = min_bound + np.array([i*GRID_SIZE, j*GRID_SIZE, 0])
            crop_max_bound = min_bound + np.array([(i+1)*GRID_SIZE, (j+1)*GRID_SIZE, pc_height])

            crop_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_min_bound, max_bound = crop_max_bound)

            terrain_tile = pc.crop(crop_bbox)

            if terrain_tile.is_empty():
                continue


            # only keep tile if full size
            if tile_is_square_of_gridsize(terrain_tile, GRID_SIZE):
                # TODO: TEMP: return one tile
                translation = np.array([-i*GRID_SIZE, -j*GRID_SIZE, 0]) - min_bound
                terrain_tile.translate(translation)
                return terrain_tile
                # # TODO: TEMP: COLOR
                terrain_tile.paint_uniform_color(np.array([0,1,0]))
            else:
                terrain_tile.paint_uniform_color(np.array([1,0,0]))

            # TODO: TEMP: slight shift for visualization
            terrain_tile.translate(np.array([i*visualization_shift, j*visualization_shift, 0]))

            tiles.append(terrain_tile)

    o3d.visualization.draw_geometries(tiles)

    return


def generate_perlin_noise():
    # sample of how perlin noise is generated in tile generation code, for testing purposes
    TILE_SIZE = GRID_SIZE

    nx = round(POINTS_PER_METER * TILE_SIZE) + 1
    ny = round(POINTS_PER_METER * TILE_SIZE) + 1

    RES = 1
    LACUNARITY = 2
    OCTAVES = 8

    # get dimensions to generate perlin noise, shape must be multiple of res*lacunarity**(octaves - 1)
    shape_factor = RES*(LACUNARITY**(OCTAVES-1))
    perlin_nx = nx - (nx % shape_factor) + shape_factor
    perlin_ny = ny - (ny % shape_factor) + shape_factor

    perlin_noise = generate_fractal_noise_2d((perlin_ny, perlin_nx), (RES, RES), octaves=OCTAVES, lacunarity=LACUNARITY)
    perlin_noise = perlin_noise[:ny, :nx]

    SCALE = 2
    perlin_noise = perlin_noise * SCALE

    x = np.linspace(0, TILE_SIZE, num = nx)
    y = np.linspace(0, TILE_SIZE, num = ny)
    xv, yv = np.meshgrid(x, y)

    interpolator = interpolate.RegularGridInterpolator((x,y), perlin_noise.T) # need to transpose perlin noise for some reason, hope this doesn't fuck anything up later lol

    points_xy = np.array([xv.flatten(), yv.flatten()]).T
    z_arr = perlin_noise.flatten()
    points_3d = np.column_stack((points_xy, z_arr))
    
    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(np.array(points_3d))

    # o3d.visualization.draw_geometries([cloud])
    return points_3d, interpolator

# TODO: check integration of both techniques with perlin noise.
#  Base statistical method gives more uniform results but has some outliers in less dense areas which may cause problems
#  Alt method based on raycloudtools is more patchy, as whole elevated patches can get removed, but is more smooth in general

def extract_lowest(terrain_tile):
    # divide tile into bins of STEP_SIZE and select bottom points in each bin based on mean in that bin
    STEP_SIZE = 0.02

    min_bound = terrain_tile.get_min_bound()
    max_bound = terrain_tile.get_max_bound()

    x_range = np.arange(min_bound[0], max_bound[0], step=STEP_SIZE)
    y_range = np.arange(min_bound[1], max_bound[1], step=STEP_SIZE)

    bottom_cloud = o3d.geometry.PointCloud()
    top_cloud = o3d.geometry.PointCloud()
    
    for x in x_range:
        for y in y_range:
            # crop to current bin
            crop_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([x,y,min_bound[2]]), max_bound=np.array([x+STEP_SIZE, y+STEP_SIZE, max_bound[2]]))
            cropped_tile = terrain_tile.crop(crop_bbox)

            if cropped_tile.is_empty():
                continue

            min_bound_part = cropped_tile.get_min_bound()

            # calculate mean of cropped tile
            mean_coords, _ = cropped_tile.compute_mean_and_covariance()
            # get bottom of tile based on mean
            z_max = min_bound_part[2] + (mean_coords[2] - min_bound_part[2])*3/4

            # crop top and bottom and add to tile top and bottom
            crop_bottom_max = np.array([max_bound[0], max_bound[1], z_max])
            crop_top_min = np.array([min_bound[0], min_bound[1], z_max])
            crop_bbox_bottom = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=crop_bottom_max)
            crop_bbox_top = o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_top_min, max_bound=max_bound)
            bottom_cloud_part = cropped_tile.crop(crop_bbox_bottom)
            bottom_cloud_part.paint_uniform_color(np.array([0,1,0]))
            top_cloud_part = cropped_tile.crop(crop_bbox_top)
            top_cloud_part.paint_uniform_color(np.array([1,0,0]))

            bottom_cloud += bottom_cloud_part
            top_cloud += top_cloud_part

    return bottom_cloud, top_cloud

def extract_lowest_alt(terrain_tile, step_size):
    # divide tile into bins of step_size and select bottom points in each bin by comparing to neighbouring bins

    min_bound = terrain_tile.get_min_bound()
    max_bound = terrain_tile.get_max_bound()

    x_range = np.arange(min_bound[0], max_bound[0], step=step_size)
    y_range = np.arange(min_bound[1], max_bound[1], step=step_size)

    bottom_cloud = o3d.geometry.PointCloud()
    top_cloud = o3d.geometry.PointCloud()

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
    
    bottom_cloud = o3d.geometry.PointCloud()
    bottom_cloud.points = o3d.utility.Vector3dVector(np.array(points_bot))
    bottom_cloud.paint_uniform_color(np.array([0,1,0]))
    top_cloud = o3d.geometry.PointCloud()
    top_cloud.points = o3d.utility.Vector3dVector(np.array(points_top))
    top_cloud.paint_uniform_color(np.array([1,0,0]))
    return bottom_cloud, top_cloud, bins, bins_lowest


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

def overlay_noise(terrain_tile, noise_tile, interpolator):
    STEP_SIZE = 0.005

    bottom_pc, top_pc, bins, bins_lowest = extract_lowest_alt(terrain_tile, step_size=STEP_SIZE)

    # for each bin: get closest 4 noise points in xy:

    points_p_dim = POINTS_PER_METER*GRID_SIZE + 1

    corrected_points = []
    
    # interpolator = interpolate.interp2d(noise_tile[:,0], noise_tile[:,1], noise_tile[:,2])

    for i in range(len(bins)):
        for j in range(len(bins[0])):
            bin_center_x = i*STEP_SIZE+STEP_SIZE/2
            bin_center_y = j*STEP_SIZE+STEP_SIZE/2

            center_arr = np.array([bin_center_x, bin_center_y])

            lowest_in_tile = False
            if bins_lowest[i][j] is None:
                lowest_point = np.array(get_closest_lowest(bins_lowest, i, j, center_arr))
            else:
                lowest_point = bins_lowest[i][j]
                lowest_in_tile = True
            
            # id_x = m.floor(bin_center_x * POINTS_PER_METER)
            # id_y = m.floor(bin_center_y * POINTS_PER_METER)

            # noise_point_tl = noise_tile[id_x + id_y*points_p_dim] # indexing flattened meshgrid coordinates # TODO: check if x and y are right or opposite
            # noise_point_bl = noise_tile[id_x + 1 + id_y*points_p_dim]
            # noise_point_tr = noise_tile[id_x + (id_y+1)*points_p_dim]
            # noise_point_br = noise_tile[id_x + 1 + (id_y+1)*points_p_dim]

            # square_points = np.vstack((noise_point_tl, noise_point_bl, noise_point_tr, noise_point_br))

            # weights = 1/np.linalg.norm((square_points[:,:2] - np.tile(center_arr, (4,1))), axis=1)

            # height = np.dot(weights, square_points[:,2])/ np.sum(weights)

            height = interpolator((bin_center_x, bin_center_y))
            height_correction = height - lowest_point[2]
            corrected_point = lowest_point + np.array([0, 0, height_correction])

            if lowest_in_tile:
                corrected_points.append(corrected_point)
    
    bottom_cloud = o3d.geometry.PointCloud()
    bottom_cloud.points = o3d.utility.Vector3dVector(np.array(corrected_points))
    bottom_cloud.paint_uniform_color(np.array([0,1,0]))

    noise_cloud = o3d.geometry.PointCloud()
    noise_cloud.points = o3d.utility.Vector3dVector(np.array(noise_tile))
    noise_cloud.paint_uniform_color(np.array([1,0,0]))

    o3d.visualization.draw_geometries([bottom_cloud, noise_cloud])

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--terrain_cloud", required=True)

    args = parser.parse_args()

    if not os.path.exists(args.terrain_cloud):
        print(f"Couldn't read input dir {args.terrain_cloud}!")
        return
    
    terrain_tile = preprocess_terrain(args.terrain_cloud)

    noise_tile, interpolator = generate_perlin_noise()

    overlay_noise(terrain_tile, noise_tile, interpolator)

    return

if __name__ == "__main__":
    main()