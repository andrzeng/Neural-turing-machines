import torch
import torch.nn as nn

class Memory(nn.Module):
    def __init__(self, 
                 dim: int,
                 num_memory_locations: int):
        """
            Description:
                Constructor for the Memory class

            Args:
                dim (int): Dimension of the memory
                num_memory_locations (int): Number of memory locations

            Returns:
                Memory: A Memory object
        """
        super().__init__()
        self.content = torch.randn(num_memory_locations, dim, requires_grad=True)

    def init_memory_constant(self, 
                             constant_value: float=1e-6,
                             device: str ='cpu') -> None:
        """
            Description:
                Initializes the state of the memory to be a constant value. This inductive bias
                has been experimentally shown to lead to faster convergence by (Collier and Beel, 2018)
            
            Arguments:
                constant_value (float): The initial value of every memory entry. Default is 1e-6
                device (str): The device of the memory

            Returns:
                None
        """
        with torch.no_grad():
            self.content = torch.ones_like(self.content).to(device) * constant_value
            
    def add_to_memory(self,
                      weighting: torch.Tensor,
                      add_vector: torch.Tensor) -> None:
        """
            Description:
                Perform an add operation on the memory

            Arguments:
                weighting (torch.Tensor): A normalized weighting over the memory slots. Weighing a slot more will modify it more
                add_vector (torch.Tensor): The vector to be added to each of the memory locations (in proportion to its weighting)

            Returns:
                None
        """
        self.content = self.content + torch.outer(weighting, add_vector)
        
    def erase_from_memory(self,
                          weighting: torch.Tensor,
                          erase_vector: torch.Tensor) -> None:
         """
            Description:
                Perform an erase operation on the memory

            Arguments:
                weighting (torch.Tensor): A normalized weighting over the memory slots. Weighing a slot more will modify it more
                erase_vector (torch.Tensor): The vector to be erased to each of the memory locations (in proportion to its weighting)

            Returns:
                None
        """
         weighted_erase = torch.outer(weighting, erase_vector)
         self.content = self.content * (torch.ones_like(weighted_erase) - weighted_erase)     
    
    