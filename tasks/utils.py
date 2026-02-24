import numpy as np
import random
from copy import deepcopy


def hierarchical_binary_patterns(dim=64, n1=4, n2=4, n3=4, p1=.1, p2=.1):
    n = n1 * n2 * n3
    relation_matrix = create_block_matrix(n, n1) + create_block_matrix(n, n1*n2) + np.eye(n)
    A = np.round(np.random.uniform(size=(n1, dim)))
    Aij = np.zeros((n1, n2, dim))
    Aijk = np.zeros((n1, n2, n3, dim))
    for i in range(n1):
        for j in range(n2):
            Aij[i, j] = random_flip(p1, deepcopy(A[i]))
            for k in range(n3):
                Aijk[i,j,k] = random_flip(p2, deepcopy(Aij[i, j]))
    data = Aijk.reshape((-1, dim))
    return relation_matrix, data


def random_flip(p, vector):
    """
    Flips p percent of the entries in a binary-valued vector.

    :param p: float, percentage of entries to flip (between 0 and 100)
    :param vector: list or numpy array of binary values (0s and 1s)
    :return: modified vector with p percent of entries flipped
    """
    if not 0 <= p <= 1:
        raise ValueError("Percentage p must be between 0 and 100")

    num_to_flip = int(np.round(len(vector) * p))
    indices_to_flip = random.sample(range(len(vector)), num_to_flip)

    for index in indices_to_flip:
        vector[index] = 1 - vector[index]  # Flip the value (0->1, 1->0)

    return vector


def create_block_matrix(m, n):
    """
    Creates an m x m block matrix with n blocks of ones along the diagonal.

    :param m: The size of the matrix (number of rows and columns).
    :param n: The number of blocks of ones along the diagonal.
    :return: An m x m numpy array representing the block matrix.

    # Example usage
    m = 6
    n = 2
    block_matrix = create_block_matrix(m, n)
    print(block_matrix)
    """
    if n > m:
        raise ValueError("Number of blocks cannot be greater than the matrix size.")

    # Initialize an m x m matrix of zeros
    matrix = np.zeros((m, m))

    # Calculate the size of each block
    block_size = m // n

    # Fill diagonal blocks with ones
    for i in range(n):
        start_index = i * block_size
        end_index = start_index + block_size
        matrix[start_index:end_index, start_index:end_index] = np.ones((block_size, block_size))

    return matrix
