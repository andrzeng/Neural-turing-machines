import torch.nn as nn
import torch
from memory import Memory
import torch.nn.functional as F

class NTMHead(nn.Module):
    def __init__(self, 
                 head_type: str,
                 dim_input: int,
                 dim_memory: int, 
                 num_memory_slots: int,
                 possible_shift_radius: int,
                 ):
        super().__init__()

        #self.initial_weighting = nn.Parameter((torch.randn(num_memory_slots)*100).softmax(dim=0), requires_grad=True)
        self.initial_weighting = nn.Parameter(torch.ones(num_memory_slots), requires_grad=True)
        assert head_type == 'read' or head_type == 'write', 'the head\'s type must either be \'read\' or \'write\''
        self.head_type = head_type
        self.possible_shift_radius = possible_shift_radius
        self.split_sizes = [dim_memory, 1, 1, 2*possible_shift_radius+1, 1]
        self.input_to_weighting_factors = nn.Linear(dim_input, sum(self.split_sizes))
        
        if(head_type == 'write'):
            self.input_to_erase_vector = nn.Linear(dim_input, dim_memory)
            self.input_to_add_vector = nn.Linear(dim_input, dim_memory)

    def read_from_memory(self, 
                         w_t: torch.Tensor,
                         memory: Memory):
        return w_t @ memory.content

    def get_weighting(self, 
                input_from_controller: torch.Tensor,
                memory: Memory,
                previous_weighting: torch.Tensor) -> torch.Tensor:
        
        key_vector, key_strength, interpolation_gate, shift_weighting, sharpening_factor = self.input_to_weighting_factors(input_from_controller).split(self.split_sizes)
        
        # key_strength = key_strength.sigmoid()
        key_strength = key_strength
        interpolation_gate = interpolation_gate.softmax(dim=0)
        shift_weighting = shift_weighting.softmax(dim=0)
        #sharpening_factor = sharpening_factor.sigmoid() + 1
        sharpening_factor = sharpening_factor.relu() + 1
        content_based_weighting = F.cosine_similarity(memory.content, key_vector, dim=1)
        content_based_weighting = F.softmax(key_strength*content_based_weighting, dim=0)
        gated_weighting = (content_based_weighting * interpolation_gate) + (1 - interpolation_gate) * previous_weighting
        shifted_weighting = torch.zeros_like(gated_weighting)
        
        for index in range(shifted_weighting.shape[0]):
            sum = 0
            for index2 in range(self.possible_shift_radius*2 + 1):
                wrapped_index = (index - self.possible_shift_radius + index2) % len(gated_weighting)
                sum += gated_weighting[wrapped_index] * shift_weighting[index2]
            
            shifted_weighting[index] = sum
        
        sharpened_weighting = (shifted_weighting ** sharpening_factor) / (shifted_weighting ** sharpening_factor).sum()
        
        return sharpened_weighting
    
    def forward(self, 
                weighting: torch.Tensor,
                memory: Memory,
                input_from_controller: torch.Tensor=None
                )-> torch.Tensor:
        
        if(self.head_type == 'read'):
            read_vector = weighting @ memory.content
            return read_vector
        
        elif(self.head_type == 'write'):
            assert input_from_controller is not None, "You must pass the controller\'s output when calling a write head's forward function."
            
            erase_vector = self.input_to_erase_vector(input_from_controller)
            memory.erase_from_memory(weighting, erase_vector)

            add_vector = self.input_to_add_vector(input_from_controller)
            memory.add_to_memory(weighting, add_vector)
            