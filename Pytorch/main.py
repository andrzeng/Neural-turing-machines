import torch.nn as nn
import torch
import random
import matplotlib.pyplot as plt
from NTM import FFNTM

def num_params(model: nn.Module):
    total = 0
    for param in model.parameters():
        total += param.numel()
    return total

if __name__ == '__main__':  

    torch.manual_seed(11)
    random.seed(12)
    model = FFNTM(dim_memory=6,
                num_memory_locations=3,
                dim_external_input=6,
                dim_controller_output=6,
                dim_NTM_output=6,
                num_read_heads=2,
                num_write_heads=2,
                possible_shift_radius=1)
    #model.load_state_dict(torch.load("checkpoints/NTM_2_epoch_80.pt"))
    print(f'Number of parameters: {num_params(model)}')
    
    loss_fn = nn.MSELoss()
    delimiter = torch.Tensor([-1,-1,-1,-1,-1,-1])
    losses = []

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98855309465)
    batch_size=16
    for epoch in range(100):
        batch_loss = 0
        for batch_num in range(batch_size):
            
            pattern = torch.Tensor([random.randint(0,1) for i in range(6)])
            num_copies = random.randint(2,10)
            #num_copies=30
            target = pattern.repeat(num_copies).reshape(num_copies, 6)

            model.memory.init_memory_constant(0)
            output, read_outputs, read_weightings, write_weightings = model.forward(external_input=pattern,
                                                                                past_read_head_outputs=None,
                                                                                past_read_weightings=None,
                                                                                past_write_weightings=None,)

            output, read_outputs, read_weightings, write_weightings = model.forward(external_input=delimiter,
                                                                                    past_read_head_outputs=read_outputs,
                                                                                    past_read_weightings=read_weightings,
                                                                                    past_write_weightings=write_weightings)

            outputs = []
            for c_index in range(num_copies):
                output, read_outputs, read_weightings, write_weightings = model.forward(external_input=torch.zeros_like(pattern),
                                                                                        past_read_head_outputs=read_outputs,
                                                                                        past_read_weightings=read_weightings,
                                                                                        past_write_weightings=write_weightings)
                
                outputs.append(output.unsqueeze(0))

            cat_output = torch.cat(outputs, dim=0)
            loss = loss_fn(cat_output, target)
            if(batch_num == 0):
                print(f'Output:\n{cat_output}\ntarget:\n{target}\n')
            loss.backward()
            #print(model.read_heads[0].input_to_weighting_factors.weight.grad)
            #print(model.read_heads[0].initial_weighting)
            batch_loss += loss.item()

        optimizer.step()
        #scheduler.step()
        optimizer.zero_grad()
        
        print(f'epoch {epoch}, loss is {batch_loss/batch_size}')
        #if(epoch % 10 == 0):
        #    torch.save(model.state_dict(), f'checkpoints/NTM_2_epoch_{epoch}.pt')
        losses.append(batch_loss/batch_size)

    plt.plot(losses)
    plt.savefig('losses.png')