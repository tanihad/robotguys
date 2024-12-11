import random
import math
import matplotlib.pyplot as plt
import numpy as np
import heapq
import pickle
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------- Node Class -----------------------------
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.neighbors = []
        self.parent = None
        self.g = float('inf')  # Cost from start to this node
        self.h = float('inf')  # Heuristic estimate from this node to goal
        self.f = float('inf')  # Total cost (f = g + h)

    def __lt__(self, other):
        return self.f < other.f
    
    def reset(self):
        self.parent = None
        self.g = float('inf')
        self.h = float('inf')
        self.f = float('inf')

# ----------------------------- Circle Class -----------------------------
class Circle:
    def __init__(self, center, radius):
        self.center = center  # (x, y) tuple
        self.radius = radius

    def contains_point(self, point):
        return np.hypot(self.center[0] - point[0], self.center[1] - point[1]) <= self.radius

    def overlaps(self, other_circle):
        return np.hypot(self.center[0] - other_circle.center[0],
                       self.center[1] - other_circle.center[1]) < (self.radius + other_circle.radius + 0.1)

# ----------------------------- Utility Functions -----------------------------

def distance_np(points1, points2):
    """Calculate Euclidean distance between two sets of points."""
    return np.linalg.norm(points1 - points2, axis=1)

def is_collision_free(node, obstacles):
    """Check if a node is collision-free (not inside any obstacle)."""
    point = np.array([node.x, node.y])
    for obstacle in obstacles:
        if obstacle.contains_point(point):
            return False
    return True

def is_collision_free_path(node1, node2, obstacles, n=25):
    """Check if the path between two nodes is collision-free by interpolating points."""
    t_values = np.linspace(0, 1, n + 2)[1:-1]
    interp_points = np.outer(1 - t_values, [node1.x, node1.y]) + np.outer(t_values, [node2.x, node2.y])
    for point in interp_points:
        for obstacle in obstacles:
            if obstacle.contains_point(point):
                return False
    return True

def generate_random_node(x_range, y_range):
    """Generate a random node within the specified ranges."""
    x = random.uniform(x_range[0], x_range[1])
    y = random.uniform(y_range[0], y_range[1])
    return Node(x, y)

def build_prm(num_samples, x_range, y_range, obstacles, k_neighbors, start_node, end_node):
    """
    Build a Probabilistic Roadmap (PRM).

    Args:
        num_samples: Number of random samples to generate.
        x_range: Tuple (min_x, max_x) defining the x-axis range.
        y_range: Tuple (min_y, max_y) defining the y-axis range.
        obstacles: List of Circle objects representing obstacles.
        k_neighbors: Number of nearest neighbors to connect.
        start_node: Start Node.
        end_node: Goal Node.

    Returns:
        List of all nodes in the PRM.
    """
    nodes = [start_node, end_node]
    for _ in range(num_samples):
        node = generate_random_node(x_range, y_range)
        if is_collision_free(node, obstacles):
            nodes.append(node)

    # Extract coordinates for KDTree
    coords = np.array([[node.x, node.y] for node in nodes])
    tree = KDTree(coords)

    # Function to find neighbors and connect nodes
    def connect_node(i):
        node = nodes[i]
        distances, indices = tree.query([node.x, node.y], k=k_neighbors + 1)
        neighbors = []
        for idx in indices[1:]:  # Skip the node itself
            neighbor = nodes[idx]
            if is_collision_free_path(node, neighbor, obstacles):
                neighbors.append(neighbor)
        return neighbors

    # Use ThreadPoolExecutor for parallel connections
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(connect_node, i): i for i in range(len(nodes))}
        for future in as_completed(futures):
            i = futures[future]
            neighbors = future.result()
            nodes[i].neighbors = neighbors

    return nodes

def find_path(start, goal, nodes):
    """
    Find a path from start to goal using the A* algorithm.

    Args:
        start: Start Node.
        goal: Goal Node.
        nodes: List of all nodes in the PRM.

    Returns:
        List of nodes representing the path, or None if no path is found.
    """
    for node in nodes:
        node.reset()

    open_list = []
    closed_set = set()

    start.g = 0
    start.h = np.hypot(start.x - goal.x, start.y - goal.y)
    start.f = start.g + start.h
    heapq.heappush(open_list, (start.f, start))

    while open_list:
        current_f, current = heapq.heappop(open_list)
        if current in closed_set:
            continue
        closed_set.add(current)

        if current == goal:
            # Reconstruct path
            path = []
            while current:
                path.append(current)
                current = current.parent
            path.reverse()
            return path

        for neighbor in current.neighbors:
            if neighbor in closed_set:
                continue

            tentative_g = current.g + np.hypot(current.x - neighbor.x, current.y - neighbor.y)

            if tentative_g < neighbor.g:
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = np.hypot(neighbor.x - goal.x, neighbor.y - goal.y)
                neighbor.f = neighbor.g + neighbor.h
                heapq.heappush(open_list, (neighbor.f, neighbor))
    return None

def generate_random_path(start, goal, nodes):
    """
    Generate a random path by selecting intermediate nodes.

    Args:
        start: Start Node.
        goal: Goal Node.
        nodes: List of all nodes in the PRM.

    Returns:
        List of nodes representing the total path, or None if no path is found.
    """
    inter_nodes = [start]
    if len(nodes) > 2:
        # Select 1 to 3 random intermediate nodes
        num_inters = random.randint(1, 3)
        inter_nodes.extend(random.sample(nodes[2:], num_inters))  # Exclude start and goal
    inter_nodes.append(goal)
    
    total_path = []
    for i in range(len(inter_nodes) - 1):
        path_segment = find_path(inter_nodes[i], inter_nodes[i+1], nodes)
        if path_segment is None:
            return None
        if i > 0:
            # Avoid duplicating nodes
            total_path.extend(path_segment[1:])
        else:
            total_path.extend(path_segment)
    return total_path

def generate_random_paths(start, goal, nodes, num_paths=1):
    """
    Generate multiple random paths.

    Args:
        start: Start Node.
        goal: Goal Node.
        nodes: List of all nodes in the PRM.
        num_paths: Number of paths to generate.

    Returns:
        List of paths, where each path is a list of nodes.
    """
    paths = []
    for _ in range(num_paths):
        path = generate_random_path(start, goal, nodes)
        if path is not None:
            paths.append(path)
    return paths

def compute_path_distance(path1, path2):
    """
    Compute the distance between two paths.
    This is a placeholder implementation that computes the sum of Euclidean distances
    between corresponding points. You may replace this with a more sophisticated metric.

    Args:
        path1: First path as a NumPy array of (x, y) tuples.
        path2: Second path as a NumPy array of (x, y) tuples.

    Returns:
        Float representing the distance between the two paths.
    """
    # Ensure both paths have the same length
    min_length = min(len(path1), len(path2))
    path1 = path1[:min_length]
    path2 = path2[:min_length]
    return np.sum(np.linalg.norm(path1 - path2, axis=1))

def is_same_homotopy(p1, p2, obstacles):
    """
    Check if two paths are in the same homotopy class.

    Args:
        p1: First path as a list of Nodes.
        p2: Second path as a list of Nodes.
        obstacles: List of Circle objects representing obstacles.

    Returns:
        Tuple (bool, list) where the bool indicates if they are in the same homotopy class,
        and the list contains connection points for visualization.
    """
    if len(p1) > 2:
        p1 = p1[1:-1]
    if len(p2) > 2:
        p2 = p2[1:-1]

    # Convert to numpy arrays for vectorization
    p1_coords = np.array([[node.x, node.y] for node in p1])
    p2_coords = np.array([[node.x, node.y] for node in p2])

    # Find nearest neighbors between p1 and p2
    tree_p2 = KDTree(p2_coords)
    distances_p1, indices_p1 = tree_p2.query(p1_coords, k=1)
    collision_free_p1 = np.all([
        is_collision_free_path(Node(*p1_coords[i]), Node(*p2_coords[indices_p1[i]]), obstacles)
        for i in range(len(p1_coords))
    ])

    tree_p1 = KDTree(p1_coords)
    distances_p2, indices_p2 = tree_p1.query(p2_coords, k=1)
    collision_free_p2 = np.all([
        is_collision_free_path(Node(*p2_coords[i]), Node(*p1_coords[indices_p2[i]]), obstacles)
        for i in range(len(p2_coords))
    ])

    connections = []
    for i in range(len(p1_coords)):
        connections.append((p1_coords[i][0], p1_coords[i][1],
                            p2_coords[indices_p1[i]][0], p2_coords[indices_p1[i]][1]))
    for i in range(len(p2_coords)):
        connections.append((p2_coords[i][0], p2_coords[i][1],
                            p1_coords[indices_p2[i]][0], p1_coords[indices_p2[i]][1]))

    return collision_free_p1 or collision_free_p2, connections

def generate_random_circles(num_obstacles, x_range, y_range, radius_range, safe_zone, start, corridor_width=1.0):
    """
    Generate random non-overlapping circular obstacles.

    Args:
        num_obstacles: Number of obstacles to generate.
        x_range: Tuple (min_x, max_x) defining the x-axis range.
        y_range: Tuple (min_y, max_y) defining the y-axis range.
        radius_range: Tuple (min_radius, max_radius) for obstacle sizes.
        safe_zone: List of two tuples defining the rectangular safe zone [(x1, y1), (x2, y2)].
        start: Start Node.
        goal: Goal Node.
        corridor_width: Width of the safe corridor around the straight line from start to goal.

    Returns:
        List of Circle objects representing obstacles.
    """
    obstacles = []
    max_attempts = 10000  # Increased attempts for higher success

    # Define a corridor around the straight line from start to goal
    corridor = {
        'start': (start.x, start.y),
        #'goal': (goal.x, goal.y),
        'width': corridor_width
    }

    def is_in_corridor(x, y):
        x1, y1 = corridor['start']
        #x2, y2 = corridor['goal']
        if x1 == x2 and y1 == y2:
            return False
        numerator = abs((y2 - y1)*x - (x2 - x1)*y + x2*y1 - y2*x1)
        denominator = math.hypot(y2 - y1, x2 - x1)
        distance_to_line = numerator / denominator
        return distance_to_line <= (corridor['width'] / 2)

    attempts = 0
    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        radius = random.uniform(*radius_range)
        x = random.uniform(x_range[0] + radius, x_range[1] - radius)
        y = random.uniform(y_range[0] + radius, y_range[1] - radius)
        new_circle = Circle((x, y), radius)

        # Check for overlap with existing obstacles
        if any(new_circle.overlaps(existing) for existing in obstacles):
            continue

        # Check if the new circle covers the start or goal
        if not is_collision_free(start, [new_circle]) :
            continue

        # Check if the new circle is inside the safe zone
        if safe_zone[0][0] <= x <= safe_zone[1][0] and safe_zone[0][1] <= y <= safe_zone[1][1]:
            continue

        # Ensure the circle does not block the corridor
        # if is_in_corridor(x, y):
        #     continue

        obstacles.append(new_circle)

    if len(obstacles) < num_obstacles:
        print(f"Warning: Only placed {len(obstacles)} out of {num_obstacles} obstacles due to space constraints.")

    return obstacles

# ----------------------------- Main Execution -----------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    # random.seed(42)
    # np.random.seed(42)

    # Define start and goal nodes
    start = Node(5, 5)
    #goal = Node(9, 9)

    # Parameters
    radius_range = (1, 1.5)      # Range of obstacle radii
    x_range = (-1, 11)              # X-axis range
    y_range = (-1, 11)              # Y-axis range
    safe_zone = [(2, 2), (3, 3)]   # Safe zone where no obstacles are placed
    num_obstacles = 100             # Number of obstacles to generate
    corridor_width = 0.0           # Width of the safe corridor

    # Generate circular obstacles
    obstacles = generate_random_circles(
        num_obstacles=num_obstacles,
        x_range=x_range,
        y_range=y_range,
        radius_range=radius_range,
        safe_zone=safe_zone,
        start=start,
        #goal=goal,
        corridor_width=corridor_width
    )

    # Build the Probabilistic Roadmap
    # nodes = build_prm(
    #     num_samples=2500,
    #     x_range=x_range,
    #     y_range=y_range,
    #     obstacles=obstacles,
    #     k_neighbors=15,  # Reduced from 100 to 15
    #     start_node=start,
    #     end_node=goal
    # )

    # Generate paths
    #paths = generate_random_paths(start, goal, nodes, num_paths=2)

    # Visualization
    plt.figure(figsize=(10, 10))
    plt.plot(start.x, start.y, "go", markersize=10, label="Start")
    #plt.plot(goal.x, goal.y, "ro", markersize=10, label="Goal")

    # Plot obstacles
    for obstacle in obstacles:
        circle = plt.Circle(obstacle.center, obstacle.radius, color='gray', fill=True, alpha=0.5)
        plt.gca().add_patch(circle)

    # format_paths = []
    # if len(paths) > 0:
    #     colors = ['b', 'm', 'c', 'y']  # Colors for multiple paths
    #     for idx, path in enumerate(paths):
    #         if path is None:
    #             continue
    #         path_coords = [(node.x, node.y) for node in path]
    #         format_paths.append(path_coords)
    #         xs, ys = zip(*path_coords)
    #         plt.plot(xs, ys, color=colors[idx % len(colors)], linewidth=2, label=f'Path {idx+1}')
    #
    #     # Compute and print path distances
    #     if len(format_paths) >= 2:
    #         path1_np = np.array(format_paths[0])
    #         path2_np = np.array(format_paths[1])
    #         path_dist = compute_path_distance(path1_np, path2_np)
    #         print("PATH DISTANCE: ", path_dist)
    #
    #         # Check homotopy
    #         p1, p2 = paths[0], paths[1]
    #         h, connections = is_same_homotopy(p1, p2, obstacles)
    #         print("SAME HOMOTOPY: ", h)
    #         # Visualize connections (optional)
    #         for conn in connections:
    #             plt.plot([conn[0], conn[2]], [conn[1], conn[3]], color='green', linestyle='--', linewidth=0.5)
    #
    # print("ALL PATHS:")
    # for idx, path in enumerate(format_paths):
    #     print(f"Path {idx+1}: {path}")

    # Save paths to a pickle file
    # with open('paths.pkl', 'wb') as file:
    #     pickle.dump(format_paths, file)

    # Final plot adjustments
    plt.xlim((0,10))
    plt.ylim((0,10))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)
    plt.axis('off')
    #plt.show()
    plt.savefig('test.png', bbox_inches='tight', pad_inches=0)

