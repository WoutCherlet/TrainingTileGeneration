import argparse
import os
import open3d as o3d
import numpy as np
from perlin_numpy import generate_fractal_noise_2d


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

    GRID_SIZE = .5 # in metres

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
                bottom_cloud, top_cloud = extract_lowest(terrain_tile)
                bottom_cloud_alt, top_cloud_alt = extract_lowest_alt(terrain_tile)
                bottom_cloud_alt.translate(np.array([0.6, 0, 0]))
                top_cloud_alt.translate(np.array([0.6,0,0]))

                o3d.visualization.draw_geometries([bottom_cloud, top_cloud, bottom_cloud_alt, top_cloud_alt])
                return
                # TODO: TEMP: COLOR
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
    POINTS_PER_METER = 10
    X_SIZE = 0.5
    Y_SIZE = 0.5

    nx = POINTS_PER_METER * X_SIZE
    ny = POINTS_PER_METER * Y_SIZE

    RES = 4
    LACUNARITY = 2
    OCTAVES = 4

    # get dimensions to generate perlin noise, shape must be multiple of res*lacunarity**(octaves - 1)
    shape_factor = RES*(LACUNARITY**(OCTAVES-1))
    perlin_nx = nx - (nx % shape_factor) + shape_factor
    perlin_ny = ny - (ny % shape_factor) + shape_factor

    perlin_noise = generate_fractal_noise_2d((perlin_ny, perlin_nx), (RES, RES), octaves=OCTAVES, lacunarity=LACUNARITY)
    perlin_noise = perlin_noise[:ny, :nx]
    SCALE = 2
    perlin_noise = perlin_noise * SCALE

    return perlin_noise

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

def extract_lowest_alt(terrain_tile):
    # divide tile into bins of STEP_SIZE and select bottom points in each bin by comparing to neighbouring bins
    STEP_SIZE = 0.01

    min_bound = terrain_tile.get_min_bound()
    max_bound = terrain_tile.get_max_bound()

    x_range = np.arange(min_bound[0], max_bound[0], step=STEP_SIZE)
    y_range = np.arange(min_bound[1], max_bound[1], step=STEP_SIZE)

    bottom_cloud = o3d.geometry.PointCloud()
    top_cloud = o3d.geometry.PointCloud()

    x_bins = len(x_range)
    y_bins = len(y_range)

    # THIS ASSUMES POINT 0,0,0 CAN NEVER BE IN TILE?
    lowest = np.ones((x_bins, y_bins, 3))*np.inf
    points_arr = np.asarray(terrain_tile.points)

    for point in points_arr:
        bin_idx = (point - min_bound) // STEP_SIZE
        idx_x = int(bin_idx[0])
        idx_y = int(bin_idx[1])
        if point[2] < lowest[idx_x][idx_y][2]:
            lowest[idx_x][idx_y] = point
        
    points_bot = []
    points_top = []
    for point in points_arr:
        bin_idx = (point - min_bound) // STEP_SIZE
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
    
    bottom_cloud = o3d.geometry.PointCloud()
    bottom_cloud.points = o3d.utility.Vector3dVector(np.array(points_bot))
    bottom_cloud.paint_uniform_color(np.array([0,1,0]))
    top_cloud = o3d.geometry.PointCloud()
    top_cloud.points = o3d.utility.Vector3dVector(np.array(points_top))
    top_cloud.paint_uniform_color(np.array([1,0,0]))
    return bottom_cloud, top_cloud

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--terrain_cloud", required=True)

    args = parser.parse_args()

    if not os.path.exists(args.terrain_cloud):
        print(f"Couldn't read input dir {args.terrain_cloud}!")
        return
    
    preprocess_terrain(args.terrain_cloud)

    return

if __name__ == "__main__":
    main()