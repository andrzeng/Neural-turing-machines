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
                 dim_controller_output: int, # same as dim_head_input
                 dim_NTM_output: int,
                 #controller: nn.Module,
                 num_read_heads: int,
                 num_write_heads: int,
                 possible_shift_radius: int,
                 ):

        super().__init__()

        self.dim_memory = dim_memory
        self.num_memory_locations = num_memory_locations
        self.dim_external_input = dim_external_input
        self.dim_controller_output = dim_controller_output
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.possible_shift_radius = possible_shift_radius

        self.memory = Memory(dim=dim_memory,
                             num_memory_locations=num_memory_locations)
        #self.memory.init_memory_constant()
        
        self.controller = nn.Linear(dim_external_input + num_read_heads * dim_memory, dim_controller_output)
        self.output_fc = nn.Linear(dim_controller_output + num_read_heads * dim_memory, dim_NTM_output)


        self.read_heads = nn.ModuleList([NTMHead(head_type='read',
                                   dim_input=dim_controller_output,
                                   dim_memory=dim_memory,
                                   num_memory_slots=num_memory_locations,
                                   possible_shift_radius=possible_shift_radius) for _ in range(num_read_heads)])
        
        self.write_heads = nn.ModuleList([NTMHead(head_type='write',
                                   dim_input=dim_controller_output,
                                   dim_memory=dim_memory,
                                   num_memory_slots=num_memory_locations,
                                   possible_shift_radius=possible_shift_radius) for _ in range(num_write_heads)])
        
    def forward(self,
                external_input: torch.Tensor,
                past_read_head_outputs: list[torch.Tensor] = None,
                past_read_weightings: list[torch.Tensor] = None,
                past_write_weightings: list[torch.Tensor] = None,
                ) -> Tuple[list, list, list]:
                
        # first concatenate
        if(past_read_head_outputs is None):
            past_read_head_outputs = [torch.zeros(self.dim_memory) for _ in range(self.num_read_heads)]
        
        input_to_controller = torch.cat([external_input, *past_read_head_outputs])
        #controller_output = self.controller(input_to_controller).sigmoid()
        controller_output = self.controller(input_to_controller)

        # First, run the read heads and collect their output
        current_read_head_weightings = []
        current_read_head_outputs = []
        for head_index, head in enumerate(self.read_heads):
            
            if(past_read_weightings is None):
                previous_w = head.initial_weighting.softmax(0)
            else:
                previous_w = past_read_weightings[head_index]
            
            weighting = head.get_weighting(input_from_controller=controller_output,
                                           memory=self.memory,
                                           previous_weighting=previous_w
                                           )
            current_read_head_weightings.append(weighting)
            
            reading = head.forward(weighting=weighting,
                                   memory=self.memory,
                                   input_from_controller=controller_output)
            current_read_head_outputs.append(reading)
        

        # Next, run the write heads. First, run the erase operation for each head, then once all the heads have run an erase operation, let them do their add op.
        
        current_write_head_weightings = []
        for head_index, head in enumerate(self.write_heads):
            if(past_write_weightings is None):
                previous_w = head.initial_weighting.softmax(0)
            else:
                previous_w = past_write_weightings[head_index]
            
            weighting = head.get_weighting(input_from_controller=controller_output,
                                           memory=self.memory,
                                           previous_weighting=previous_w
                                           )
          
            current_write_head_weightings.append(weighting)
            
        
        for head_index, head in enumerate(self.write_heads):
            weighting = current_write_head_weightings[head_index]
            head.forward(weighting=weighting,
                         memory=self.memory,
                         input_from_controller=controller_output
                         )
        
        input_to_output_fc = torch.cat([controller_output, *current_read_head_outputs])
        #NTM_output = self.output_fc(input_to_output_fc).sigmoid()
        NTM_output = self.output_fc(input_to_output_fc)
        return NTM_output, current_read_head_outputs, current_read_head_weightings, current_write_head_weightings