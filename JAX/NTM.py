import jax.numpy as jnp
from flax import linen as nn
from head import ReadHead, WriteHead
from typing import List

class NTM(nn.Module):
    dim_memory: int
    num_memory_locations: int
    dim_controller_output: int
    dim_NTM_output: int
    num_read_heads: int
    num_write_heads: int
    shift_radius: int
    batch_size: int
    dim_external_input: int

    def setup(self) -> None:
        """
            Description:
                Initialize model parameters
            Args:
                None
            Returns:
                None
        """
        self.controller = nn.Dense(self.dim_controller_output)
        self.read_heads = [ReadHead(self.dim_memory, self.num_memory_locations, self.shift_radius) for _ in range(self.num_read_heads)]
        self.write_heads = [WriteHead(self.dim_memory, self.num_memory_locations, self.shift_radius) for _ in range(self.num_write_heads)]
        self.output_linear = nn.Dense(self.dim_NTM_output)

    def __call__(self,
                 external_input: jnp.array,
                 past_readings: jnp.array,
                 past_read_w: jnp.array,
                 past_write_w: jnp.array,
                 memory: jnp.array) -> list[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
        """
            Description:
                The forward function of the NTM

            Args:
                external_input (jnp.array): External input to the NTM
                past_readings (jnp.array): Past outputs of the read heads
                past_read_w (jnp.array): Past read head weightings
                past_write_w (jnp.array): Past write head weightings
                memory (jnp.array): The memory to be operated on

            Returns:
                jnp.array: the NTM's output
                jnp.array: the modified memory
                jnp.array: the output of the read heads
                jnp.array: the weightings of the read heads
                jnp.array: the weightigns of the write heads
        """
        if(past_readings is None):
            past_readings = jnp.zeros((self.batch_size, self.num_read_heads, self.dim_memory))
        past_readings = past_readings.reshape(self.batch_size, -1)
        input_to_controller = jnp.concatenate([external_input, past_readings], axis=1)
        controller_output = self.controller(input_to_controller)
        new_read_w = []
        new_write_w = []
        read_out = []
        for index, head in enumerate(self.read_heads):
            if(past_read_w is None):
                previous_w = jnp.ones((self.batch_size, self.num_memory_locations))
                previous_w = nn.softmax(previous_w)
            else:
                previous_w = past_read_w[index]
            read, weighting = head(controller_output, memory, previous_w)
            new_read_w.append(weighting)
            read_out.append(read)

        for index, head in enumerate(self.write_heads):
            if(past_write_w is None):
                previous_w = jnp.ones((self.batch_size, self.num_memory_locations))
                previous_w = nn.softmax(previous_w)
            else:
                previous_w = past_write_w[index]

            memory, weighting = head(controller_output, memory, previous_w)
            new_write_w.append(weighting)

        input_for_output_fc = jnp.concatenate([controller_output, jnp.concatenate(read_out, axis=1)], axis=1)
        NTM_output = self.output_linear(input_for_output_fc)
        read_out = jnp.concatenate(read_out, axis=1)

        return NTM_output, memory, read_out, new_read_w, new_write_w
        