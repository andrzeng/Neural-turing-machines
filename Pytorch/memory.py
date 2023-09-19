import torch
import torch.nn as nn

class Memory(nn.Module):
    def __init__(self, 
                 dim: int,
                 num_memory_locations: int):
        super().__init__()
        self.content = torch.randn(num_memory_locations, dim, requires_grad=True)
        
    def init_memory_initial(self, device='cpu'):
        raise NotImplementedError
        self.content = self.initial_content.clone()

    def init_memory_constant(self, 
                             constant_value: float=1e-6,
                             device='cpu'):
        with torch.no_grad():
            self.content = torch.ones_like(self.content).to(device) * constant_value
    
    def init_memory_random(self,
                           device='cpu'):
         with torch.no_grad():
            self.content = torch.randn_like(self.content).to(device) * 1e-2
            
    def add_to_memory(self,
                      weighting: torch.Tensor,
                      add_vector: torch.Tensor):
        self.content = self.content + torch.outer(weighting, add_vector)
        
    def erase_from_memory(self,
                          weighting: torch.Tensor,
                          erase_vector: torch.Tensor):
         
         weighted_erase = torch.outer(weighting, erase_vector)
         self.content = self.content * (torch.ones_like(weighted_erase) - weighted_erase)     
    
    