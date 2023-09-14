import jax
import jax.numpy as jnp
from flax import linen as nn
import utils

class NTMHead(nn.Module):
    dim_memory: int
    num_memory_slots: int
    shift_radius: int

    def setup(self):
        self.input_to_weighting_factors = nn.Dense(self.dim_memory + 1 + 1 + 2*self.shift_radius+1 + 1)
        self._split_indices = utils.chunksize_to_index([self.dim_memory, 1, 1, 2*self.shift_radius+1, 1])
    
    def get_w_c(self, 
                memory, 
                key_vector, 
                key_strength):
        content_weighting = utils.cosine_similarity(memory, key_vector)
        content_weighting = nn.softmax(content_weighting * key_strength)
        return content_weighting

    def get_w_s(self, 
                w_g, 
                shift_radius, 
                conv_kernel):   
        shifted_weighting = jnp.zeros_like(w_g)
        stacked = []
        for index in range(shifted_weighting.shape[0]):
            _sum = 0
            for index2 in range(2*shift_radius+1):
                wrapped_index = (index - 1 + index2) % w_g.shape[0]
                _sum += w_g[wrapped_index] * conv_kernel[index2]
            stacked.append(_sum)
        final = jnp.stack(stacked, axis=0)
        return final

    def __call__(self,
            input_from_controller,
            memory,
            previous_weighting):      
        _combined = self.input_to_weighting_factors(input_from_controller)
        key_vector, key_strength, gate, conv_kernel, sharpening_factor, _ = jax.vmap(jnp.split, [0, None])(_combined, self._split_indices)
        gate = nn.sigmoid(gate)
        conv_kernel = nn.softmax(conv_kernel, axis=1)
        sharpening_factor = nn.relu(sharpening_factor) + 1
        w_c = self.get_w_c(memory, key_vector, key_strength)
        w_g = (w_c * gate) + (1 - gate) * previous_weighting
        w_s = jax.vmap(self.get_w_s, [0, None, 0])(w_g, self.shift_radius, conv_kernel)
        
        return w_s

class ReadHead(NTMHead):
    def setup(self):
        super().setup()
    
    def __call__(self,
                 input_from_controller,
                 memory,
                 previous_weighting):
        weighting = super().__call__(input_from_controller, memory, previous_weighting)
        return jax.vmap(jnp.matmul)(weighting, memory), weighting

class WriteHead(NTMHead):
    def setup(self):
        super().setup()
        self.input_to_edit_vectors = nn.Dense(self.dim_memory * 2)
    
    def _erase(self, memory, weighting, erase_vector):
        return memory * (jnp.ones_like(memory) - jax.vmap(jnp.outer)(weighting, erase_vector))
    
    def _add(self, memory, weighting, add_vector):
        return memory + jax.vmap(jnp.outer)(weighting, add_vector)
    
    def get_weighting(self,
                      input_from_controller,
                      memory,
                      previous_weighting):
        return super().__call__(input_from_controller, memory, previous_weighting)

    def write(self, 
              weighting,
              input_from_controller,
              memory):
        edit_vectors = self.input_to_edit_vectors(input_from_controller)
        add_vector, erase_vector = jax.vmap(jnp.split, [0, None])(edit_vectors, [self.dim_memory])
        memory = self._add(memory, weighting, add_vector)
        memory = self._erase(memory, weighting, erase_vector)
        return memory
    
    def __call__(self,
                 input_from_controller,
                 memory,
                 previous_weighting):
        weighting = super().__call__(input_from_controller, memory, previous_weighting)
        edit_vectors = self.input_to_edit_vectors(input_from_controller)
        add_vector, erase_vector = jax.vmap(jnp.split, [0, None])(edit_vectors, [self.dim_memory])
        memory = self._add(memory, weighting, add_vector)
        memory = self._erase(memory, weighting, erase_vector)
        return memory, weighting