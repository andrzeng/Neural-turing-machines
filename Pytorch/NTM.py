import torch
import torch.nn as nn
from memory import Memory
from head import NTMHead
from typing import Tuple

class FFNTM(nn.Module):
    def __init__(self,
                 dim_memory: int,
                 num_memory_locations: int,
                 dim_external_input: int,
                 dim_controller_output: int,
                 dim_NTM_output: int,
                 num_read_heads: int,
                 num_write_heads: int,
                 shift_radius: int,
                 ):
        """
            Description:
                This is the constructor function for a Neural Turing Machine with a single-layer feedforward controller network.

            Args:
                dim_memory (int): The dimension of each slot in the NTMs's memory
                num_memory_locations (int): The number of slots in the NTM's memory
                dim_external_input (int): The dimension of the input to the NTM. 
                dim_controller_output (int): The dimension of the controller network's output, 
                                        which is fed to the read and write heads.
                dim_NTM_output (int): The dimension of the output of the entire NTM.
                num_read_heads (int): The number of read heads
                num_write_heads (int): The number of write heads
                shift_radius (int): The max possible shift radius during location-based addressing
            
            Returns:
                FFNTM: A Neural Turing Machine with the specified parameters
        """
        
        super().__init__()

        self.dim_memory = dim_memory
        self.num_memory_locations = num_memory_locations
        self.dim_external_input = dim_external_input
        self.dim_controller_output = dim_controller_output
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.shift_radius = shift_radius

        self.memory = Memory(dim_memory, num_memory_locations)
        self.controller = nn.Linear(dim_external_input + num_read_heads * dim_memory, dim_controller_output)
        self.output_fc = nn.Linear(dim_controller_output + num_read_heads * dim_memory, dim_NTM_output)
        self.read_heads = nn.ModuleList([NTMHead('read', dim_controller_output, dim_memory, num_memory_locations, shift_radius) for _ in range(num_read_heads)])
        self.write_heads = nn.ModuleList([NTMHead('write', dim_controller_output, dim_memory, num_memory_locations, shift_radius) for _ in range(num_write_heads)])
        
    def forward(self,
                external_input: torch.Tensor,
                past_read_head_outputs: list[torch.Tensor] = None,
                past_read_weightings: list[torch.Tensor] = None,
                past_write_weightings: list[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, list, list, list]:
        """
            Description:
                This is the forward pass function of the Neural Turing Machine

            Args:  
                external_input (torch.Tensor): The external input to the NTM
                past_read_head_outputs (list[torch.Tensor]): The outputs of each read head from the last forward pass.
                                                            If this is the first pass, then they are None
                past read_weightings (list[torch.Tensor]): The weightings over memory of each read head from the last forward pass.
                                                            If this is the first pass, then they are None
                past_write_weightings (list[torch.Tensor]): The weightings over memory of each write head from the last forward pass.
                                                            If this is the first pass, then they are None
            
            Returns:
                torch.Tensor: the final output of the NTM
                List[torch.Tensor]: the output of each read head, collected into a list 
                List[torch.Tensor]: the weighting of each read head, collected into a list 
                List[torch.Tensor]: the output of each write head, collected into a list 
        
        """

        # first concatenate
        if(past_read_head_outputs is None):
            past_read_head_outputs = [torch.zeros(self.dim_memory) for _ in range(self.num_read_heads)]
        
        input_to_controller = torch.cat([external_input, *past_read_head_outputs])
        controller_output = self.controller(input_to_controller)

        # First, run the read heads and collect their output
        current_read_head_weightings = []
        current_read_head_outputs = []
        for head_index, head in enumerate(self.read_heads):
            
            if(past_read_weightings is None):
                previous_w = head.initial_weighting.softmax(0)
            else:
                previous_w = past_read_weightings[head_index]
            
            weighting = head.get_weighting(controller_output, self.memory, previous_w)
            current_read_head_weightings.append(weighting)
            reading = head.forward(weighting, self.memory, controller_output)
            current_read_head_outputs.append(reading)
        
        # Next, run the write heads. First, run the erase operation for each head, then once all the heads have run an erase operation, let them do their add op.
        current_write_head_weightings = []
        for head_index, head in enumerate(self.write_heads):
            if(past_write_weightings is None):
                previous_w = head.initial_weighting.softmax(0)
            else:
                previous_w = past_write_weightings[head_index]
            
            weighting = head.get_weighting(controller_output, self.memory, previous_w)
            current_write_head_weightings.append(weighting)
            
        for head_index, head in enumerate(self.write_heads):
            weighting = current_write_head_weightings[head_index]
            head.forward(weighting, self.memory, controller_output)
        
        input_to_output_fc = torch.cat([controller_output, *current_read_head_outputs])
        NTM_output = self.output_fc(input_to_output_fc)
        return NTM_output, current_read_head_outputs, current_read_head_weightings, current_write_head_weightings