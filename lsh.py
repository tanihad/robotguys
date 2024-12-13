import numpy as np
import torch
from model import EmbeddingNetworkModule,ImageMixtureDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def retrieve():
    checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
    model = EmbeddingNetworkModule.load_from_checkpoint(checkpoint)
    model.eval()

    # choose your trained nn.Module
    encoder = model.encoder
    encoder.eval()

    #
    embedding_size = 16
    lsh = RandomProjectionHash(embeddim=embedding_size)

    # embed images
    # with torch.no_grad():
    #     embeddings = model.embedding(image_batch)

    data = ImageMixtureDataset("imgs", "masks", "goal_directions.npy")
    dataloader = DataLoader(data, batch_size=1, shuffle=False)

    train_indices = range(0, int(len(data) * 0.9))
    test_indices = range(int(len(data) * 0.9), len(data))

    #dataloader for train and test

    train_loader = DataLoader(data, batch_size=1, sampler=train_indices)
    test_loader = DataLoader(data, batch_size=1, sampler=test_indices)

    with torch.no_grad():
        for i, (image1, _, _, _, _, _) in enumerate(train_loader):
            image1 = image1.to(model.device)
            npembedding = model.embedding(image1).cpu().numpy().flatten()
            lsh.insert(npembedding, data_id=i)

    with torch.no_grad():
        for i, (image1, _, _, _, _, _) in enumerate(test_loader):
            image1 = image1.to(model.device)
            npembedding = model.embedding(image1).cpu().numpy().flatten()

            similar_indices = lsh.query(npembedding, top_k=3)
            print("Index ",i)
            print("Similar index: ",similar_indices)

    # with torch.no_grad():
    #     for i, (image1, _, _, _, _, _) in enumerate(dataloader):
    #         image1 = image1.to(model.device)
    #         npembedding = model.embedding(image1).cpu().numpy().flatten()#Make np array
    #         lsh.insert(npembedding, data_id= i)
            #lsh.insert(npembedding, data_id=whatever theo returns here)

        #for i in DataLoader(enumerate):





class RandomProjectionHash:
    def __init__(self, embeddim: int, numprojections=16, projectdim=8, table_size=100):
        self.projections = numprojections
        self.projectdim = projectdim
        self.table_size = table_size
        self.db = {}

        self.projection_matrices = [np.random.randn(embeddim, projectdim) for _ in range(numprojections)]

    def hash_vector(self, vector):
        vector_hash = []
        for proj in self.projection_matrices:
            projection_result = np.dot(vector, proj)
            sign_result = np.sign(np.sum(projection_result))

            if sign_result > 0:
                vector_hash.append('1')
            else:
                vector_hash.append('0')

        return ''.join(vector_hash)

    def _hash_to_index(self, vector_hash):
        return int(vector_hash, 2) % self.table_size

    def insert(self, vector, data_id):
        print("THis is my vector ",vector)
        bvector_hash = self.hash_vector(vector)
        print("binary ",bvector_hash)
        index = self._hash_to_index(bvector_hash)
        print("the index ",index)
        if index not in self.db:
            self.db[index] = []

        self.db[index].append((bvector_hash, data_id, vector))

    def query(self, vector, top_k=5):
        query_hash = self.hash_vector(vector)
        index = self._hash_to_index(query_hash)
        matches = []

        if index in self.db:
            for _, data_id, stored_vector in self.db[index]:
                similarity = self.cosine_similarity(vector, stored_vector)
                print("Cosine similarity between ",vector,"and ","stored_vector",similarity)
                matches.append((similarity, data_id))

        return [data_id for _, data_id in sorted(matches, reverse=True)[:top_k]]

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))



def visualize_lsh_embeddings_with_images(train_embeddings, test_embeddings, train_indices, test_indices, similar_indices, train_images, test_images, title="LSH Embedding Visualization"):
    """
    Visualizes embeddings and their LSH-based similarity with image annotations.

    Parameters:
    - train_embeddings: numpy array of shape (num_train, 2) with train embeddings in 2D.
    - test_embeddings: numpy array of shape (num_test, 2) with test embeddings in 2D.
    - train_indices: list of train indices.
    - test_indices: list of test indices.
    - similar_indices: list of lists, where similar_indices[i] contains the indices of train embeddings similar to test_embeddings[i].
    - train_images: list of images corresponding to train embeddings.
    - test_images: list of images corresponding to test embeddings.
    - title: Title for the plot.
    """
    plt.figure(figsize=(12, 10))

    # Plot train embeddings
    train_embeddings = np.array(train_embeddings)
    plt.scatter(train_embeddings[:, 0], train_embeddings[:, 1], c='blue', label='Train Embeddings', alpha=0.7)

    # Plot test embeddings
    test_embeddings = np.array(test_embeddings)
    plt.scatter(test_embeddings[:, 0], test_embeddings[:, 1], c='red', label='Test Embeddings', alpha=0.7)

    # Function to add images to the plot
    def add_image_to_plot(image, coords, zoom=0.1):
        imagebox = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(imagebox, coords, frameon=False)
        plt.gca().add_artist(ab)

    # Annotate test points and connect to similar train points
    for i, test_idx in enumerate(test_indices):
        test_point = test_embeddings[i]
        plt.annotate(f"T{test_idx}", (test_point[0], test_point[1]), color='red')
        add_image_to_plot(test_images[i], test_point)

        for sim_idx in similar_indices[i]:
            train_point = train_embeddings[train_indices.index(sim_idx)]
            plt.plot([test_point[0], train_point[0]], [test_point[1], train_point[1]], 'g--', alpha=0.5)

    # Annotate train points
    for i, train_idx in enumerate(train_indices):
        train_point = train_embeddings[i]
        plt.annotate(f"Tr{train_idx}", (train_point[0], train_point[1]), color='blue')
        add_image_to_plot(train_images[i], train_point)

    plt.title(title)
    plt.legend()
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.grid(True)
    plt.show()

# Example usage:
# Assuming train_embeddings, test_embeddings are 2D arrays and similar_indices is generated from LSH.
# train_embeddings = np.random.rand(10, 2)  # Replace with real embeddings
# test_embeddings = np.random.rand(5, 2)   # Replace with real embeddings
# train_indices = list(range(len(train_embeddings)))
# test_indices = list(range(len(test_embeddings)))
# similar_indices = [[0, 2, 4], [1, 3, 5], ...]  # Replace with actual similar indices
# train_images = [np.random.rand(10, 10, 3) for _ in range(len(train_embeddings))]  # Replace with real images
# test_images = [np.random.rand(10, 10, 3) for _ in range(len(test_embeddings))]  # Replace with real images
# visualize_lsh_embeddings_with_images(train_embeddings, test_embeddings, train_indices, test_indices, similar_indices, train_im


if __name__ == '__main__':
    test = RandomProjectionHash(embeddim=2, numprojections=16, projectdim=8, table_size=10)
    vector_1 = np.array([1., 2.])
    vector_2 = np.array([2., 3.])
    vector_3 = np.array([3., 4.])
    vector_4 = np.array([5., 6.])
    test.insert(vector_1, "1")
    test.insert(vector_2, "2")
    test.insert(vector_3, "3")
    test.insert(vector_4, "4")
    result = test.query(vector_1, top_k=3)
    print("Top 3 ", result)