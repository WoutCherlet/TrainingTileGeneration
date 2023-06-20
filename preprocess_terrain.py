import argparse
import os
import open3d as o3d
import numpy as np

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

    # TODO: we can get more tiles out of same plot if we rotate longest side to be parallel to xy

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
                terrain_tile.paint_uniform_color(np.array([0,1,0]))
            else:
                terrain_tile.paint_uniform_color(np.array([1,0,0]))

            # TODO: TEMP: slight shift for visualization
            terrain_tile.translate(np.array([i*visualization_shift, j*visualization_shift, 0]))

            tiles.append(terrain_tile)

    o3d.visualization.draw_geometries(tiles)

    return


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