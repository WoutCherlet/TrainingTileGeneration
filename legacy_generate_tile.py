'''

Old code for generate tile, might come in handy when bugs occur to simplify

'''


def generate_random_tree_height(alpha=2, beta=2):
    Z_MAX = 1 # max tree height deviation from 0

    beta_sample =  np.random.beta(a=5,b=5) # beta distribution looks kind off like a normal distribution but with values bounded in [0,1]
    z_target = (2*beta_sample - 1)*Z_MAX # recentre around zero and scale with Z_MAX
    return z_target



def place_tree_in_line(name, tree_mesh, plot_mesh, collision_manager, trees):
    assert False, "don't use anymore, only for debug"

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
    assert False, "don't use anymore, only for debug"
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

    # maximal absolute height value
    z_target = generate_random_tree_height()

    initial_translation = np.array([max_x_row-min_x_tree, max_y_plot-min_y_tree, z_target-min_z_tree])
    bbox_transform = trimesh.transformations.translation_matrix(initial_translation)

    # print(f"Placing {name} at height {z_target}")

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

def influence_function(index, total_points):
    x = index/total_points
    # return 1/(1 + (x/(1-x))**2)  # reverse S shape, seems to give too much of "pedestal" kind of form
    # return -(x-1)**3 # simple exponential-like curve 
    # NOTE: use exponential? will never get 0 so no div by 0 errors eg e^-4*x is practically equivalent to above
    return (x-1)**2


def trunk_height_influence_map(min_x, min_y, ny, nx, points_per_meter, trunk_locations):
    # NOTE: don't use this, use convex hull function instead
    
    # NOTE: x and y are reversed here, deprecated
    assert False, "x and y are still reversed in this function, change implementation first"

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
    # NOTE: x and y are reversed here, deprecated
    assert False, "x and y are still reversed in this function, change implementation first"

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




def terrain2mesh(terrain_cloud, decimation_factor = 5):
    tri = Delaunay(terrain_cloud.point.positions.numpy()[:,:2])

    terrain_mesh = o3d.t.geometry.TriangleMesh(vertex_positions=terrain_cloud.point.positions, triangle_indices=o3d.core.Tensor(tri.simplices))

    # perform decimation
    decimated_mesh = terrain_mesh.simplify_quadric_decimation(target_reduction = 1.0 - 1.0/2)

    # visualization for debug
    # terrain_mesh.translate([40,0,0])
    # decimated_mesh.translate([80,0,0])
    # o3d.visualization.draw([terrain_cloud, terrain_mesh, decimated_mesh])

    return decimated_mesh

