import jax
import jax.numpy as jnp

@jax.jit
def cosine_similarity(memories, key_vectors, eps=1e-8):
    def single_cs(memory, key_vector):
        similarity_score = memory @ key_vector
        norm_products = jnp.linalg.norm(memory, axis=-1) * jnp.linalg.norm(key_vector)
        return similarity_score / (norm_products + jnp.ones_like(norm_products) * eps)

    return jax.vmap(single_cs)(memories, key_vectors)

def chunksize_to_index(chunk_sizes):
    indices = []
    for index, _ in enumerate(chunk_sizes):
        if(index == 0):
            indices.append(chunk_sizes[index])
        else:
            indices.append(indices[index-1] + chunk_sizes[index])
    return indices

def num_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
