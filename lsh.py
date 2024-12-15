import numpy as np
import torch
from model import EmbeddingNetworkModule,ImageMixtureDataset, EmbeddingNetwork, device
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


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
        #print("THis is my vector ",vector)
        bvector_hash = self.hash_vector(vector)
        #print("binary ",bvector_hash)
        index = self._hash_to_index(bvector_hash)
        #print("the index ",index)
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
                #print("Cosine similarity between ",vector,"and ","stored_vector",similarity)
                matches.append((similarity, data_id))

        return [data_id for _, data_id in sorted(matches, key=lambda tup: tup[0], reverse=True)[1:(top_k+1)]]

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))



def retrieve():
    checkpoint = "./epoch=611-step=86292.ckpt"
    embedding_size = 16
    model_p = EmbeddingNetwork()
    model = EmbeddingNetworkModule.load_from_checkpoint(checkpoint, embedding=model_p)
    model.eval()

    # choose your trained nn.Module
    embedding = model.embedding
    embedding.eval()

    #
    lsh = RandomProjectionHash(embeddim=embedding_size)

    # embed images
    # with torch.no_grad():
    #     embeddings = model.embedding(image_batch)

    train_data = ImageMixtureDataset("imgs", "masks", "goal_directions.npy", data_range=(0, 9000))
    test_data = ImageMixtureDataset("imgs", "masks", "goal_directions.npy", data_range=(9000, 10000-2))

    train_loader = DataLoader(train_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)

    with torch.no_grad():
        for i, (image1, mask1, dir1, _, _, _) in enumerate(train_loader):
            image_npy = image1.cpu().numpy()
            mask_npy = mask1.cpu().numpy()
            dir_npy = dir1.cpu().numpy()

            image1 = image1.to(device)
            dir1 = dir1.to(device)

            npembedding = model.embedding(image1, dir1).cpu().numpy().flatten()
            lsh.insert(npembedding, data_id=(image_npy, mask_npy, dir_npy))

    original_images = []
    similar_images_arrays = []

    with torch.no_grad():
        for i, (image1, mask1, dir1, _, _, _) in enumerate(test_loader):
            image_npy = image1.cpu().numpy()
            mask_npy = mask1.cpu().numpy()
            dir_npy = dir1.cpu().numpy()
            original_images.append((image_npy, mask_npy, dir_npy))

            image1 = image1.to(device)
            dir1 = dir1.to(device)
            npembedding = model.embedding(image1, dir1).cpu().numpy().flatten()

            similar_images = lsh.query(npembedding, top_k=3)

            similar_images_arrays.append(similar_images)

            print("Index ",i)


    for original_img, similar_imgs in zip(original_images, similar_images_arrays):
        print(original_img)
        original_img, original_mask, direct = original_img
        # Create a new figure for each iteration
        fig, axes = plt.subplots(1, len(similar_imgs) + 1, figsize=(15, 5))

        # Plot the original image with caption 'Original'
        axes[0].imshow(original_mask.squeeze(), cmap='gray')  # Use 'cmap' for grayscale images if needed
        axes[0].axis('off')
        axes[0].set_title('Original')

        height, width = original_img.squeeze().shape
        center_x = width / 2
        center_y = height / 2

        axes[0].quiver(center_x, center_y, 7*direct.squeeze()[0], 7*direct.squeeze()[1], angles='xy', scale_units='xy', scale=1, color='red')

        # Plot each similar image with caption 'Similar'
        for i, similar_img in enumerate(similar_imgs):
            similar_img, similar_mask, direct = similar_img
            axes[i + 1].imshow(similar_mask.squeeze(), cmap='gray')
            axes[i + 1].axis('off')
            axes[i + 1].set_title('Similar')

            axes[i + 1].quiver(center_x, center_y, 7*direct.squeeze()[0], 7*direct.squeeze()[1], angles='xy', scale_units='xy', scale=1, color='red')

        # Display the plot
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    retrieve()
