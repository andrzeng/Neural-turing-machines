import jax
import jax.numpy as jnp

@jax.jit
def cosine_similarity(memories: jnp.array, 
                      key_vectors: jnp.array, 
                      eps: float = 1e-8) -> jnp.array:
    """
        Description: 
            Calculates the cosine similarity of a key vector with a memory array

        Args:
            memories (jnp.array): The memory array to calculate cosine similarity over
            key_vectors (jnp.array): Key vectors to be compared with entries in the memory
            eps (float): A small value added to the denominator to avoid division by zero

        Returns:
            jnp.array: the cosine similarity score between every memory entry and every key vector
    """
    def _single_cs(memory, key_vector):
        similarity_score = memory @ key_vector
        norm_products = jnp.linalg.norm(memory, axis=-1) * jnp.linalg.norm(key_vector)
        return similarity_score / (norm_products + jnp.ones_like(norm_products) * eps)

    return jax.vmap(_single_cs)(memories, key_vectors)

def chunksize_to_index(chunk_sizes: list) -> list:
    """
        Description: 
            Given a size-wise list of blocks, return the cumulative size sum at each block

        Args:
            chunk_sizes (list): List of chunk sizes

        Returns:
            list: The cumulative chunk sizes
    """
    indices = []
    for index, _ in enumerate(chunk_sizes):
        if(index == 0):
            indices.append(chunk_sizes[index])
        else:
            indices.append(indices[index-1] + chunk_sizes[index])
    return indices

def num_params(params: dict) -> int:
    """
        Description: 
            Count the number of params in a model

        Args:
            params (dict): A dictionary of model parameters

        Returns:
            int: the number of parameters
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
