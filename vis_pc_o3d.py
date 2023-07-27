import open3d as o3d
import glob
import os
import numpy as np

# def read_trees(pc_dir):

#     trees = {}
#     for file in sorted(glob.glob(os.path.join(pc_dir, "*.ply"))):
#         pc = o3d.io.read_point_cloud(file)
#         name = os.path.basename(file)[:-4]
#         trees[name] = pc
#     return trees

# pc_dir = "/home/wcherlet/data/singletrees/Wytham_sample/"

# trees = read_trees(pc_dir)
# y_offset = 0
# merged_cloud = None
# n_trees = 0
# trees_vis = 3
# for tree in trees:
#     pc = trees[tree]

#     min_x, min_y, min_z = pc.get_min_bound()
#     _, max_y, _ = pc.get_max_bound()

#     pc = pc.translate(np.array([-min_x, y_offset - min_y, -min_z]))
#     y_offset += (max_y - min_y)

#     if merged_cloud is None:
#         merged_cloud = pc
#     else:
#         merged_cloud += pc

#     n_trees += 1
#     if n_trees >= trees_vis:
#         break


# lookat = np.array([ 10.616374969482422, 33.670360565185547, 14.409182548522949 ])
# zoom = 0.4
# up = np.array([0,0,1])
# front = np.array([1,0,0])


# o3d.visualization.draw_geometries([merged_cloud], lookat=lookat, up=up, zoom=zoom, front=front)

# o3d.io.write_point_cloud("trees.ply", merged_cloud)



# pc = "/home/wcherlet/data/singletrees/Wytham_sample/synthetic_tiles/downsampled/Tile_2.ply"
pc = "/home/wcherlet/InstSeg/training_synthesis/trees_merged_isolated.ply"
# pc = "/home/wcherlet/InstSeg/training_synthesis/assets/terrain_tiles.ply"
# pc = "/home/wcherlet/InstSeg/training_synthesis/assets/terrain.ply"

pcloud = o3d.io.read_point_cloud(pc)
print(pcloud.get_min_bound(), pcloud.get_max_bound())

pcloud = pcloud.voxel_down_sample(voxel_size=0.05)

# consistent view

lookat = np.array([ 7.6225957895274199, 14.289404290648148, -22.451114791887342 ])
zoom = 0.8
up = np.array([ -0.2747098001314619, -0.26396491448757653, 0.92458479850757813 ])
front = np.array([ 0.69268020219658266, 0.61258727329786333, 0.38069800377515833 ])

# for terrain_tiles
# zoom = 0.4
lookat = np.array([4, 4, -1])

o3d.visualization.draw_geometries([pcloud], lookat=lookat, up=up, zoom=zoom, front=front)

