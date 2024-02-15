import os
import open3d as o3d
import numpy as np
import laspy
import matplotlib.pyplot as plt

def read_txt(file):
    arr = np.loadtxt(file, dtype=float, skiprows=1)
    o3d_pc = o3d.t.geometry.PointCloud()
    o3d_pc.point.positions = o3d.core.Tensor(arr[:,:3])
    return o3d_pc

def read_las(file):
    point_cloud = laspy.read(file)
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()

    o3d_pc = o3d.t.geometry.PointCloud()
    o3d_pc.point.positions = o3d.core.Tensor(points)
    return o3d_pc

def viz_pc(pc):
    bbox = pc.to_legacy().get_axis_aligned_bounding_box()
    o3d.visualization.draw_geometries([pc.to_legacy(), bbox])
    return

def merge_all(pointclouds):
    pointcloud = pointclouds[0]
    for pc in pointclouds[1:]:
        pointcloud += pc
    return pointcloud

def viz_trees_on_tile(tile_file, trees_folder):

    trees = []

    for file in os.listdir(trees_folder):
        tree = read_txt(os.path.join(trees_folder, file))

        color = np.random.choice(range(256), size=3)/ 256

        legacy_tree = tree.to_legacy()
        legacy_tree.paint_uniform_color(color)

        trees.append(legacy_tree)

    tile = read_txt(tile_file)
    legacy_tile = tile.to_legacy()

    trees.append(legacy_tile)

    o3d.visualization.draw_geometries(trees)

def read_tiles(folder):
    pcs = []

    for file in os.listdir(folder):
        if file[:4] == "tile":
            pc = read_txt(os.path.join(folder, file))

            pc = pc.voxel_down_sample(voxel_size=0.20)

            legacy_pc = pc.to_legacy()
            bbox = legacy_pc.get_axis_aligned_bounding_box()

            pcs.append(legacy_pc)
            # pcs.append(bbox)

    return pcs

    


def litchfield_full_plot(understory_tiles, trees_parent_folder):

    pcs = read_tiles(understory_tiles)

    merged_understory = merge_all(pcs)

    max_bound = merged_understory.get_max_bound()
    min_bound = merged_understory.get_min_bound()

    print(f"Max_bound of plot: {max_bound}")
    print(f"Min_bound of plot: {min_bound}")
    print(f"Dimension of plot: {max_bound-min_bound}")

    o3d.visualization.draw_geometries(pcs)


def get_xy_view(understory_tiles):
    tilenames = [f for f in sorted(os.listdir(understory_tiles)) if f[-3:] == 'txt']
    
    x = []
    y = []
    for tilename in tilenames:
        file = os.path.join(understory_tiles, tilename)

        pc = read_txt(file)

        center = (pc.get_max_bound().numpy() + pc.get_min_bound().numpy())/2

        x.append(center[0])
        y.append(center[1])

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(tilenames):
        ax.annotate(txt, (x[i], y[i]))

    plt.show()


    

def main():

    FOLDER = "/media/wcherlet/SSD WOUT/2019_ElizaSteffen_thesis/Understorey/OK_TILES_SEPT"
    TILE_FILE = os.path.join(FOLDER, "tile_0_-20_SEP_US_OK.txt")

    PC_FOLDER = "/media/wcherlet/SSD WOUT/2019_ElizaSteffen_thesis/Bomen/tile_0_-20_BOMEN/September"

    # viz_trees_on_tile(TILE_FILE, PC_FOLDER)

    # get_xy_view(FOLDER)
    litchfield_full_plot(FOLDER, PC_FOLDER)

    return

if __name__ == "__main__":
    main()