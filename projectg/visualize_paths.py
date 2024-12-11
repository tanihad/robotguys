from generate_paths import Node, Rect
from path_similarity import alternate_kmedoids
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
   #### GET CLUSTERS ####
   paths = np.load('paths.pkl', allow_pickle=True)
   sim_matrix = np.load('paths_sim_matrix.npy')
   medoids, clusters, cost = alternate_kmedoids(sim_matrix, 8, 100)
   
   #### VISUALIZATION ####
   start = Node(1, 1)
   goal = Node(9, 9)
   obstacles = [Rect((2, 2), 2, 2), Rect((2, 6), 2, 2), Rect((6, 6), 2, 2), Rect((6, 2), 2, 2)]

   plt.plot([start.x, goal.x], [start.y, goal.y], "ro")

   cluster_colors = ['lightgrey', 'lightcoral', 'bisque', 'gold', 'limegreen', 'turquoise', 'cornflowerblue', 'violet', 'lightpink']
   medoid_colors = ['grey', 'red', 'orange', 'goldenrod', 'green', 'teal', 'blue', 'm', 'deeppink']

   for m in range(len(clusters)):
      ax = plt.subplot(int(len(clusters)/2), 2, m+1)
      ax.set_xlim(0, 10)
      ax.set_ylim(0, 10)
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.legend()
      # plot paths in cluster
      for path_index in clusters[m]:
         path_np = np.array(paths[path_index])
         plt.plot(path_np[:,0], path_np[:,1], color=cluster_colors[m], marker='o')
      # plot medoid
      medoid_np = np.array(paths[medoids[m]])
      plt.plot(medoid_np[:,0], medoid_np[:,1], color=medoid_colors[m], marker='o')
      # plot obstacles
      for i in obstacles:
         rectangle = plt.Rectangle((i.llx, i.lly), i.width, i.height, linewidth=2, edgecolor='red', facecolor='none')
         plt.gca().add_patch(rectangle)

   plt.gca().set_aspect('equal', adjustable='box')
   plt.show()