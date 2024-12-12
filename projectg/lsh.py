import numpy as np


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