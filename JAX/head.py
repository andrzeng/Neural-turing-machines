import jax
import jax.numpy as jnp
from flax import linen as nn
import utils

class NTMHead(nn.Module):
    dim_memory: int
    num_memory_slots: int
    shift_radius: int

    def setup(self) -> None:
        """
            Description:
                Initialize parameters for the NTMHead superclass
                
            Args:
                None

            Returns:     
                None
        """
        self.input_to_weighting_factors = nn.Dense(self.dim_memory + 1 + 1 + 2*self.shift_radius+1 + 1)
        self._split_indices = utils.chunksize_to_index([self.dim_memory, 1, 1, 2*self.shift_radius+1, 1])
    
    def get_w_c(self, 
                memory: jnp.array, 
                key_vector: jnp.array, 
                key_strength: float) -> jnp.array:
        """
            Description:
                Get the content based weighting
            
            Args:
                memory (jnp.array): The memory array
                key_vector (jnp.array): The content to compare to the memory locations to produce the weighting
                key_strength (float): The amount which to weigh closer matches in the memory
        
            Returns:
                jnp.array: The content-based weighting
        """
        content_weighting = utils.cosine_similarity(memory, key_vector)
        content_weighting = nn.softmax(content_weighting * key_strength)
        return content_weighting

    def get_w_s(self, 
                w_g: jnp.array, 
                shift_radius: int, 
                conv_kernel: jnp.array) -> jnp.array:  
        """
            Description:
                Get the shift weighting

            Args:
                w_g (jnp.array): The content-based weighting
                shift_radius (int): The possible shift radius
                conv_kernel (jnp.array): The shift kernel to apply to the weighting
                
            Returns:
                jnp.array: The shifted weighting
        """ 
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
            input_from_controller: jnp.array,
            memory: jnp.array,
            previous_weighting: jnp.array) -> jnp.array:      
        """
            Description:
                Obtain the final weighting over memory slots, combining the location- and content-based weightings

            Args:
                input_from_controller (jnp.array): The input from the controller network
                memory (jnp.array): The memory array
                previous_weighting (jnp.array): The previous weighting of the head

            Returns:
                jnp.array: The finalized weighting
        """
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
        """
            Description:
                Set up the read head
            Args:
                None
            Returns:
                None
        """
        super().setup()
    
    def __call__(self,
                 input_from_controller: jnp.array,
                 memory: jnp.array,
                 previous_weighting: jnp.array) -> jnp.array:
        """
            Description:
                Forward function for a read head; Obtains the read vector

            Args:
                input_from_controller (jnp.array): The input from the controller network
                memory (jnp.array): The memory array
                previous_weighting (jnp.array): The previous weighting of the head

            Returns:
                jnp.array: The read vector
        """
        weighting = super().__call__(input_from_controller, memory, previous_weighting)
        return jax.vmap(jnp.matmul)(weighting, memory), weighting

class WriteHead(NTMHead):
    def setup(self):
        """
            Description:
                Set up the write head
            Args:
                None
            Returns:
                None
        """
        super().setup()
        self.input_to_edit_vectors = nn.Dense(self.dim_memory * 2)
    
    def _erase(self, 
               memory: jnp.array, 
               weighting: jnp.array, 
               erase_vector: jnp.array) -> jnp.array:
        """
            Description:
                Erase a vector from memory, given the memory and a weighting
            Args:
                memory (jnp.array): the memory array
                weighting (jnp.array): a normalized weighting over memory slots
                erase_vector (jnp.array): the vector to erase

            Returns:
                jnp.array: the modified memory
        """
        return memory * (jnp.ones_like(memory) - jax.vmap(jnp.outer)(weighting, erase_vector))
    
    def _add(self, 
             memory: jnp.array, 
             weighting: jnp.array, 
             add_vector: jnp.array) -> jnp.array:
        """
            Description:
                Add a vector to memory, given the memory and a weighting
            Args:
                memory (jnp.array): the memory array
                weighting (jnp.array): a normalized weighting over memory slots
                add_vector (jnp.array): the vector to add

            Returns:
                jnp.array: the modified memory
        """
        return memory + jax.vmap(jnp.outer)(weighting, add_vector)
    
    def get_weighting(self,
                      input_from_controller: jnp.array,
                      memory: jnp.array,
                      previous_weighting: jnp.array) -> jnp.array:
        """
            Description:
                Calculate the weighting over the memory

            Args:
                input_from_controller (jnp.array): the input from the controller network
                memory (jnp.array): the memory array
                previous_weighting (jnp.array): the head's previous weighting

            Returns:
                jnp.array: the obtained weighting
        """
        return super().__call__(input_from_controller, memory, previous_weighting)

    def write(self, 
              weighting: jnp.array,
              input_from_controller: jnp.array,
              memory: jnp.array) -> jnp.array:
        """
            Description:
                Perform a write operation using this head to a memory array

            Args:
                weighting (jnp.array): the weighting over memory locations
                input_from_controller (jnp.array): the input from the controller network
                memory (jnp.array): the memory arrray

            Returns:
                jnp.array: the modified memory
        """
        edit_vectors = self.input_to_edit_vectors(input_from_controller)
        add_vector, erase_vector = jax.vmap(jnp.split, [0, None])(edit_vectors, [self.dim_memory])
        memory = self._add(memory, weighting, add_vector)
        memory = self._erase(memory, weighting, erase_vector)
        return memory
    
    def __call__(self,
                 input_from_controller: jnp.array,
                 memory: jnp.array,
                 previous_weighting: jnp.array) -> jnp.array:
        """
            Description:
                Write to memory using this write head

            Args:
                input_from_controller (jnp.array): The input from the controller network
                memory (jnp.array): The memory array
                previous_weighting (jnp.array): The previous weighting of the head

            Returns:
                jnp.array: The modified memory array
                weighting: the current weighting of this head
        """
        weighting = super().__call__(input_from_controller, memory, previous_weighting)
        edit_vectors = self.input_to_edit_vectors(input_from_controller)
        add_vector, erase_vector = jax.vmap(jnp.split, [0, None])(edit_vectors, [self.dim_memory])
        memory = self._add(memory, weighting, add_vector)
        memory = self._erase(memory, weighting, erase_vector)
        return memory, weighting