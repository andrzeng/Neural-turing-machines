import jax
import jax.numpy as jnp
import jax.random as random
from NTM import NTM
import utils

BATCH_SIZE = 16
INPUT_LENGTH = 10
NUM_MEMORY_LOCATIONS = 2
NUM_READ_HEADS=3
NUM_WRITE_HEADS=3
DIM_CONTROLLER_OUTPUT=6
DIM_MEMORY=6
SHIFT_RADIUS=3
NUM_BATCHES=1000

MIN_SEQ_LEN = 2
MAX_SEQ_LEN = 10

DELIMITER = jnp.ones((INPUT_LENGTH,)) * -1
DELIMITER = jnp.expand_dims(DELIMITER, axis=0).repeat(BATCH_SIZE, axis=0)

def forward_and_backward(params, inp, target, num_repeats, lr=1e-3):
    def loss_fn(params):
        memory = jnp.ones((BATCH_SIZE, NUM_MEMORY_LOCATIONS, DIM_MEMORY)) * 1e-6
        NTM_output, memory, read_out, new_read_w, new_write_w = model.apply(params, inp, None, None, None, memory)
        NTM_output, memory, read_out, new_read_w, new_write_w = model.apply(params, DELIMITER, read_out, new_read_w, new_write_w, memory)
        outputs = []
        for _ in range(num_repeats):
            NTM_output, memory, read_out, new_read_w, new_write_w = model.apply(params, jnp.zeros_like(inp), read_out, new_read_w, new_write_w, memory)
            outputs.append(NTM_output)

        outputs = jnp.concatenate(outputs, axis=1)
        difference = outputs - target
        loss = jax.vmap(jnp.inner)(difference, difference)
        return loss.mean()
    
    loss = loss_fn(params)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params)
    params = jax.tree_util.tree_map(lambda param, grad: param - grad * lr, params, grads)

    return params, loss


if __name__ == "__main__":
    model = NTM(dim_memory=DIM_MEMORY,
                num_memory_locations=NUM_MEMORY_LOCATIONS,
                dim_controller_output=DIM_CONTROLLER_OUTPUT,
                dim_NTM_output=INPUT_LENGTH,
                num_read_heads=NUM_READ_HEADS,
                num_write_heads=NUM_WRITE_HEADS,
                shift_radius=SHIFT_RADIUS,
                batch_size=BATCH_SIZE,
                dim_external_input=INPUT_LENGTH)
    key = random.PRNGKey(0)
    params = model.init(key, 
                        external_input=jnp.zeros((BATCH_SIZE, INPUT_LENGTH,)), 
                        past_readings=None, past_read_w=None, 
                        past_write_w=None, 
                        memory=jnp.zeros((BATCH_SIZE,NUM_MEMORY_LOCATIONS,DIM_MEMORY)))
    print(f"Beginning training. The model has {utils.num_params(params)} parameters")

    for i in range(NUM_BATCHES):
        num_repeats = 3
        key, _ = random.split(key)
        inp = random.randint(key, (BATCH_SIZE, INPUT_LENGTH), 0, 2)
        target = inp.repeat(num_repeats, axis=1)
        params, loss = forward_and_backward(params, inp, target, num_repeats, lr=1e-3)
        print(f"Batch {i} loss = {loss}")
        #if(i % 10 == 0):
        #    jnp.save(f"batch_{i}", params)

