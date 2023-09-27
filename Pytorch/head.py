import torch.nn as nn
import torch
from memory import Memory
import torch.nn.functional as F
from typing import Optional

class NTMHead(nn.Module):
    def __init__(self, 
                 head_type: str,
                 dim_input: int,
                 dim_memory: int, 
                 num_memory_slots: int,
                 shift_radius: int,
                 ):
        """
            Description: 
                The constructor for the head of an NTM

            Args:
                head_type (str): Either 'read' or 'write'. Specifies the type of the head
                dim_input (int): The input dimension of the head
                dim_memory (int): The dimension of the memory that the head interacts with
                num_memory_slots (int): The number of memory slots available to the head
                shift_radius (int): The possible shift radius of the head

            Returns:
                An NTMHead object
        """

        super().__init__()

        self.initial_weighting = nn.Parameter(torch.ones(num_memory_slots), requires_grad=True)
        self.head_type = head_type
        self.shift_radius = shift_radius
        self.split_sizes = [dim_memory, 1, 1, 2*shift_radius+1, 1]
        self.input_to_weighting_factors = nn.Linear(dim_input, sum(self.split_sizes))
        
        if(head_type == 'write'):
            self.input_to_erase_vector = nn.Linear(dim_input, dim_memory)
            self.input_to_add_vector = nn.Linear(dim_input, dim_memory)

    def read_from_memory(self, 
                         w_t: torch.Tensor,
                         memory: Memory) -> torch.Tensor:
        """
            Description:
                Perform a read operation on the passed memory, given a weight vector over the memory locations

            Args:
                w_t (torch.Tensor): A probability-vector weighting over the memory locations
                memory (Memory): A Memory object to be read from

            Returns:
                A torch.Tensor containing the read vector
        """
        return w_t @ memory.content

    def get_weighting(self, 
                input_from_controller: torch.Tensor,
                memory: Memory,
                previous_weighting: torch.Tensor) -> torch.Tensor:
        """
            Description:
                Get the combined content-based and location-based addressing weighting

            Args:
                input_from_controller (torch.Tensor): The controller's input which is used to create the location weighting
                memory (Memory): The memory, which will be used to create the content weighting
                previous_weighting (torch.Tensor): The weighting produced by this head in the previous step

            Returns:
                A tensor representing the combined location- and content-based weighting
        """
        key_v, key_s, gate, shift_w, sharpening = self.input_to_weighting_factors(input_from_controller).split(self.split_sizes)
        
        gate = gate.softmax(dim=0)
        shift_w = shift_w.softmax(dim=0)
        sharpening = sharpening.relu() + 1
        content_w = F.cosine_similarity(memory.content, key_v, dim=1)
        content_w = F.softmax(key_s*content_w, dim=0)
        gated_w = (content_w * gate) + (1 - gate) * previous_weighting
        shifted_w = torch.zeros_like(gated_w)
        
        for index in range(shifted_w.shape[0]):
            sum = 0
            for index2 in range(self.shift_radius*2 + 1):
                wrapped_index = (index - self.shift_radius + index2) % len(gated_w)
                sum += gated_w[wrapped_index] * shift_w[index2]
            shifted_w[index] = sum
        sharpened_weighting = (shifted_w ** sharpening) / (shifted_w ** sharpening).sum()
        
        return sharpened_weighting
    
    def forward(self, 
                weighting: torch.Tensor,
                memory: Memory,
                input_from_controller: torch.Tensor=None
                )-> Optional[torch.Tensor]:
        """
            Description:
                The forward function for the read or write head

            Args:
                weighting (torch.Tensor): The combined content and location based weighting
                memory (Memory): The memory to be read from or written to
                input_from_controller (torch.Tensor): The controller's output

            Returns:
                Returns the vector read from memory if the head is a read head;
                Otherwise does not return anything
        """
        
        if(self.head_type == 'read'):
            read_vector = weighting @ memory.content
            return read_vector
        
        elif(self.head_type == 'write'):
            erase_vector = self.input_to_erase_vector(input_from_controller)
            memory.erase_from_memory(weighting, erase_vector)
            add_vector = self.input_to_add_vector(input_from_controller)
            memory.add_to_memory(weighting, add_vector)
            