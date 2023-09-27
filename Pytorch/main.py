import torch.nn as nn
import torch
import random
import matplotlib.pyplot as plt
from NTM import FFNTM

if __name__ == '__main__':  
    torch.manual_seed(42)
    random.seed(42)
    model = FFNTM(dim_memory=6,
                num_memory_locations=3,
                dim_external_input=6,
                dim_controller_output=6,
                dim_NTM_output=6,
                num_read_heads=2,
                num_write_heads=2,
                shift_radius=1)
    
    loss_fn = nn.MSELoss()
    delimiter = torch.Tensor([-1,-1,-1,-1,-1,-1])
    losses = []

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)
    batch_size=16
    for epoch in range(100):
        batch_loss = 0
        for batch_num in range(batch_size):
            
            pattern = torch.Tensor([random.randint(0,1) for _ in range(6)])
            num_copies = random.randint(2,10)
            target = pattern.repeat(num_copies).reshape(num_copies, 6)

            model.memory.init_memory_constant(0)
            _, _, read_weightings, write_weightings = model.forward(pattern, None, None, None)
            _, _, read_weightings, write_weightings = model.forward(delimiter, read_outputs, read_weightings, write_weightings)

            outputs = []
            for c_index in range(num_copies):
                output, read_outputs, read_weightings, write_weightings = model.forward(torch.zeros_like(pattern), read_outputs, read_weightings, write_weightings)
                outputs.append(output.unsqueeze(0))

            cat_output = torch.cat(outputs, dim=0)
            loss = loss_fn(cat_output, target)
            loss.backward()
            batch_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()
        
        print(f'epoch {epoch}, loss is {batch_loss/batch_size}')
        #if(epoch % 10 == 0):
        #    torch.save(model.state_dict(), f'checkpoints/NTM_2_epoch_{epoch}.pt')
        losses.append(batch_loss/batch_size)

    plt.plot(losses)
    plt.savefig('losses.png')