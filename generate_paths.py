import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import heapq
import pickle
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def generate_random_circles(num_obstacles, x_range, y_range, radius_range, safe_zone, start):
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

def get_gaussian(circle1, circle2):
    center_distance = np.hypot(circle1.center[0] - circle2.center[0], circle1.center[1] - circle2.center[1])
    edge_distance = center_distance - (circle1.radius + circle2.radius)
    
    if edge_distance <= 1.0:
        direction = np.array([circle2.center[0] - circle1.center[0], circle2.center[1] - circle1.center[1]])
        direction /= np.linalg.norm(direction)
        midpoint = np.array(circle1.center) + direction * (circle1.radius + edge_distance / 2)
        direction2 = direction * np.array([-1, 1])
        # Scale:
        direction *= edge_distance
        direction2 *= (circle1.radius + circle2.radius) / 2
        #print("D1: ", direction)
        #print("D2: ", direction2)
        #print("COV: ", np.outer(direction, direction2))
        #print("FGHJ: ", tuple(midpoint))
        #print("FGHJ: ", (direction, direction2))
        return tuple(midpoint), (direction, direction2)
    return None

def get_all_gaussians(obstacles, goal_direction=None):
    means = []
    covs = []
    for i in range(len(obstacles)-1):
        for j in range(i+1, len(obstacles)):
            gaussian = get_gaussian(obstacles[i], obstacles[j])
            if gaussian is not None:
                mean, cov = gaussian
                to_mean = (np.array(mean)-np.array([5.0, 5.0]))
                to_mean /= np.linalg.norm(to_mean)
                if goal_direction is None or to_mean @ goal_direction > 0.7:
                    means.append(mean)
                    covs.append(cov)
    return means, covs

def random_unit_vector():
    vec = np.random.normal(size=2)
    return vec / np.linalg.norm(vec)

# ----------------------------- Main Execution -----------------------------
if __name__ == "__main__":
    # Define start and goal nodes
    start = Node(5, 5)
    #goal = Node(9, 9)

    # Parameters
    radius_range = (1, 1.5)      # Range of obstacle radii
    x_range = (-1, 11)              # X-axis range
    y_range = (-1, 11)              # Y-axis range
    safe_zone = [(2, 2), (3, 3)]   # Safe zone where no obstacles are placed
    num_obstacles = 100             # Number of obstacles to generate

    # Generate circular obstacles
    for i in range(100):
        # Generate random direction
        goal_direction = random_unit_vector()
        print("GOAL DIRECTION: ", goal_direction)

        obstacles = generate_random_circles(
            num_obstacles=random.randint(4, 30),
            x_range=x_range,
            y_range=y_range,
            radius_range=radius_range,
            safe_zone=safe_zone,
            start=start
        )

        # Get gaussians between obstacles in goal direction
        means, covs = get_all_gaussians(obstacles, goal_direction=goal_direction)

        # Visualization
        plt.figure(figsize=(10, 10))
        plt.plot(start.x, start.y, "go", markersize=10, label="Start")

        # Plot obstacles
        for obstacle in obstacles:
            circle = plt.Circle(obstacle.center, obstacle.radius, color='grey', fill=True, alpha=0.5)
            plt.gca().add_patch(circle)

        # Final plot adjustments
        plt.xlim((0,10))
        plt.ylim((0,10))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(False)
        plt.axis('off')
        #plt.show()
        pth = 'imgs/img' + str(i) + '.png'
        plt.savefig(pth, bbox_inches='tight', pad_inches=0)
        plt.clf()

        for j in range(len(means)):
            ellipse = Ellipse(
                xy=means[j],
                width=np.linalg.norm(covs[j][0]),
                height=np.linalg.norm(covs[j][1]),
                angle=np.degrees(np.arctan2(covs[j][0][1], covs[j][0][0])),
                facecolor="black",
                edgecolor="black"
            )
            #circle = plt.Circle(i, 0.5, color='r', fill=True, alpha=0.5)
            #plt.gca().add_patch(circle)
            plt.gca().add_patch(ellipse)

        plt.xlim((0,10))
        plt.ylim((0,10))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(False)
        plt.axis('off')
        #plt.show()
        pth = 'masks/mask' + str(i) + '.png'
        plt.savefig(pth, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close('all')
