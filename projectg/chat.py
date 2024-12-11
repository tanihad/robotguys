import random
import math
import matplotlib.pyplot as plt
import numpy as np
import heapq

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

class Rect:
    def __init__(self, xy, width, height):
        self.llx, self.lly = xy[0], xy[1]
        self.width = width
        self.height = height

    def contains(self, node_x, node_y):
        if self.llx <= node_x <= self.llx + self.width and self.lly <= node_y <= self.lly + self.height:
            return True
        return False

def distance(node1, node2):
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

def is_collision_free(node, obstacles):
    for obstacle in obstacles:
        if obstacle.contains(node.x, node.y):
            return False
    return True

def is_collision_free_path(node1, node2, obstacles, n=50):
    t_values = np.linspace(0, 1, n + 2)[1:-1]
    points = [Node(node1.x + t * (node2.x - node1.x), node1.y + t * (node2.y - node1.y)) for t in t_values]
    for i in points:
        if not is_collision_free(i, obstacles):
            return False
    return True

def generate_random_node(x_range, y_range):
    x = random.uniform(x_range[0], x_range[1])
    y = random.uniform(y_range[0], y_range[1])
    return Node(x, y)

def build_prm(num_samples, x_range, y_range, obstacles, k_neighbors, start_node, end_node):
    nodes = [start_node, end_node]
    for _ in range(num_samples):
        node = generate_random_node(x_range, y_range)
        if is_collision_free(node, obstacles):
            nodes.append(node)

    for node in nodes:
        neighbors = sorted(nodes, key=lambda n: distance(node, n))[:k_neighbors]
        for neighbor in neighbors:
            if neighbor != node and is_collision_free(neighbor, obstacles) and is_collision_free_path(node, neighbor, obstacles):
                node.neighbors.append(neighbor)

    return nodes

def find_path(start, goal, nodes):
    for node in nodes:
        node.g = float('inf')
        node.h = float('inf')
        node.f = float('inf')
        node.parent = None

    open_list = []
    closed_list = set()

    start.g = 0
    start.h = distance(start, goal)
    start.f = start.g + start.h
    heapq.heappush(open_list, (start.f, start))

    while open_list:
        current_f, current = heapq.heappop(open_list)
        closed_list.add(current)

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = current.parent
            path.reverse()
            return path

        for neighbor in current.neighbors:
            if neighbor in closed_list:
                continue

            tentative_g = current.g + distance(current, neighbor)

            if tentative_g < neighbor.g:
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = distance(neighbor, goal)
                neighbor.f = neighbor.g + neighbor.h
                heapq.heappush(open_list, (neighbor.f, neighbor))
    return None

def generate_safe_path(start, goal, num_waypoints, variability, radius):
    path = [start]
    for i in range(1, num_waypoints - 1):
        t = i / (num_waypoints - 1)
        x = (1 - t) * start.x + t * goal.x + random.uniform(-variability, variability)
        y = (1 - t) * start.y + t * goal.y + random.uniform(-variability, variability)
        path.append(Node(x, y))
    path.append(goal)
    safe_zones = [(node.x, node.y, radius) for node in path]
    return safe_zones

def generate_random_obstacles(num_obstacles, x_range, y_range, obstacle_size_range, safe_zones):
    obstacles = []
    for _ in range(num_obstacles):
        while True:
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            width = random.uniform(obstacle_size_range[0], obstacle_size_range[1])
            height = random.uniform(obstacle_size_range[0], obstacle_size_range[1])
            obstacle = Rect((x, y), width, height)

            # Check if the obstacle overlaps the safe zones
            is_clear = True
            for sx, sy, radius in safe_zones:
                if (x - sx) ** 2 + (y - sy) ** 2 <= radius ** 2:
                    is_clear = False
                    break

            if is_clear:
                obstacles.append(obstacle)
                break
    return obstacles

if __name__ == "__main__":
    start = Node(1, 1)
    goal = Node(9, 9)
    obstacle_size_range = (1, 2)
    x_range = (0, 10)
    y_range = (0, 10)

    # Generate a guaranteed safe path
    safe_zones = generate_safe_path(start, goal, num_waypoints=10, variability=1.5, radius=1.0)

    # Generate random obstacles
    num_obstacles = 30
    obstacles = generate_random_obstacles(num_obstacles, x_range, y_range, obstacle_size_range, safe_zones)

    # Build PRM and find path
    nodes = build_prm(500, x_range, y_range, obstacles, 10, start, goal)
    path = find_path(start, goal, nodes)

    # Visualize the environment
    plt.figure()

    # Plot safe zones
    for sx, sy, radius in safe_zones:
        circle = plt.Circle((sx, sy), radius, color='green', alpha=0.3)
        plt.gca().add_patch(circle)

    # Plot obstacles
    for obstacle in obstacles:
        rectangle = plt.Rectangle((obstacle.llx, obstacle.lly), obstacle.width, obstacle.height,
                                   linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rectangle)

    # Plot path
    if path:
        plt.plot([node.x for node in path], [node.y for node in path], 'b-', label="Path")

    # Plot start and goal
    plt.plot(start.x, start.y, "bo", label="Start")
    plt.plot(goal.x, goal.y, "go", label="Goal")

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
