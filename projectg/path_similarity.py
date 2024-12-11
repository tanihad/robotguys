import numpy as np
import matplotlib.pyplot as plt

### PATH PARAMETERIZATION METHODS ### 
def compute_pairwise_distances(path):
   return np.linalg.norm(path[1:]-path[:-1], axis=1)

def interpolate(a, b, pct):
   return ((b-a)*pct + a).reshape(1,2)

def sample_along_polyline(path, n_points):
   pdist = compute_pairwise_distances(path)
   sample_dist = np.sum(pdist)/(n_points-1)
   mod = 0
   points = path[0].reshape(1, 2)
   path = np.delete(path, 0, axis=0)
   while path.shape[0]>0:
      points = np.append(points, interpolate(points[-1], path[0], (sample_dist-mod)/pdist[0]), axis=0)
      pdist[0] -= (sample_dist-mod)
      if pdist[0]/sample_dist < 1:
         # handle numerical errors at end of path
         if path.shape[0] == 1 and not np.array_equal(points[-1],path[0]):
            points = np.append(points, path[0].reshape(1,2), axis=0)
         mod = pdist[0]
         pdist = np.delete(pdist, 0, axis=0)
         path = np.delete(path, 0, axis=0)
      else:
         mod = 0
   return points

### DISTANCE METRIC ### 
def dist(a, b):
   return np.sqrt(np.sum(np.power(a-b, 2)))

def compute_path_distance (path_1, path_2):
   path_dists = -1 * np.ones((path_1.shape[0],path_2.shape[0]))

   def path_distance(path_1, path_2):
      if path_dists[path_1.shape[0]-1, path_2.shape[0]-1] != -1:
         return path_dists[path_1.shape[0]-1, path_2.shape[0]-1]
      if path_1.shape[0] == 1 and path_2.shape[0] == 1:
         path_dists[0,0] = dist(path_1[0], path_2[0])
      elif path_1.shape[0] == 1:
         path_dists[0, path_2.shape[0]-1] = path_distance(path_1, path_2[:-1])
      elif path_2.shape[0] == 1:
         path_dists[path_1.shape[0]-1] = path_distance(path_1[:-1], path_2)
      else:
         path_dists[path_1.shape[0]-1, path_2.shape[0]-1] = min(path_distance(path_1[:-1], path_2), path_distance(path_1, path_2[:-1]), path_distance(path_1[:-1], path_2[:-1])) + dist(path_1[-1], path_2[-1])
      return path_dists[path_1.shape[0]-1, path_2.shape[0]-1]

   return path_distance(path_1, path_2)

### PATH SIMILARITY ###
def eval_path_similarity(path_1, path_2):
   return compute_path_distance(sample_along_polyline(np.array(path_1), 50), sample_along_polyline(np.array(path_2), 50))

def compute_similarity_matrix(paths):
   sim_matrix = np.ones((len(paths), len(paths)))*-1
   for i in range(len(paths)):
      for j in range(len(paths)):
         if sim_matrix[i,j] == -1:
            sim_matrix[i,j] = eval_path_similarity(paths[i],paths[j])
            sim_matrix[j,i] = sim_matrix[i,j]
      print("computed similarity for ", i, " and all j")
   np.save('paths_sim_matrix.npy', sim_matrix)
   return sim_matrix

### CLUSTERING ###
def init_random_kmedoids(paths, k):
   return (np.random.rand(k) * len(paths)).astype(int)

def init_alternate_kmedoids(sim_matrix, k):
   v = np.zeros(sim_matrix.shape[0])
   for j in range(sim_matrix.shape[0]):
      for i in range(sim_matrix.shape[0]):
         v[j]+=sim_matrix[i,j]/np.sum(sim_matrix[i])
   v_idx_sorted = np.argsort(v)
   #TODO: potentially add way to prevent redundant medoids in case of identical paths?
   return v_idx_sorted[:k]

def get_cost(sim_matrix, clusters, medoids):
   cost = 0
   for m in range(len(clusters)):
      cost+=np.sum(sim_matrix[clusters[m], medoids[m]])
   return cost

def alternate_kmedoids(sim_matrix, k, max_iter):
   iter = 0
   # Step 1: initial medoids and clustering
   medoids = init_alternate_kmedoids(sim_matrix, k)
   clusters = [[] for i in range(k)]
   for i in range(sim_matrix.shape[0]):
      closest = np.argmin(sim_matrix[i,medoids])
      clusters[closest].append(i)
   
   last_cost = -1
   cost = get_cost(sim_matrix, clusters, medoids)
   print("Medoids initialized: ", medoids)

   while cost != last_cost and iter < max_iter:
      last_cost = cost
      # Step 2: update medoids
      for m in range(len(clusters)):
         min_sum = 100000
         min_idx = -1
         for i in clusters[m]:
            sum = np.sum(sim_matrix[i,clusters[m]])
            if sum < min_sum:
               min_sum = sum
               min_idx = i
         medoids[m] = min_idx if min_idx != -1 else medoids[m]
      # Step 3: assign objects to medoids
      clusters = [[] for i in range(k)]
      for i in range(sim_matrix.shape[0]):
         closest = np.argmin(sim_matrix[i,medoids])
         clusters[closest].append(i)
      cost = get_cost(sim_matrix, clusters, medoids)
      print("Iteration ", iter, " medoids: ", medoids)
      iter += 1
   if (iter == max_iter):
      print("Terminated: max iterations reached")
   return medoids, clusters, cost

if __name__ == "__main__":
   paths = np.load('paths.pkl', allow_pickle=True)
   paths2 = np.load('paths2.pkl', allow_pickle=True)

   sim_matrix = np.load('paths_sim_matrix.npy')

   costs = []
   for k in range(10):
      medoids, clusters, cost = alternate_kmedoids(sim_matrix, k, 100)
      costs.append(cost)

   plt.plot(np.arange(10)+1, costs)