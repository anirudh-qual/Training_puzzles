import asyncio
from typing import List

from lib import Dist, Model
from utils import (
    get_model_theoretical_best_time,
    get_training_stats,
    write_chrome_trace,
)
from utils import get_global_batch_list

async def part2_training_loop(model: Model) -> Model:
    """
    Implement distributed training using Pipeline parallelism.

    Split (shard) the layers and optimizer states equally between GPUs (ranks).
    Pass the full set of batches for activations and gradient activations between layers.
    Additionally, explore how passing only a subset of batches for activations and gradient activations between layers affects performance (in terms of memory usage and e2e time)

    Coordinate the forward and backward passes across multiple GPUs in a pipelined manner. Verify the pipelined nature of the training using `visualize.py`.

    Use `model.send` and `model.receive` to pass data between different ranks. See `lib.py` for more details on these functions.
    """
    
    weights, opt_states, activations, grad_activations, grad_weights = model.storage()
    rank=model.rank
    layer_per_rank = model.num_layers // model.world_size
    start = rank * layer_per_rank
    end = start + layer_per_rank

    for l in range(start, end):
        weights[l],opt_states[l]=model.load_weights(l)
    
    for microbatch in get_global_batch_list(model.global_batch_size,model.global_batch_size):
        
        if rank!=0:
            activations[start] = await model.receive(source=rank-1)
        else:
            activations[start] = model.get_input_activation(microbatch)
        

        for l in range(start, end):
            activations[l + 1] = model.forward(l, activations[l], weights[l])

        if rank != model.world_size-1:
            await model.send(rank+1,activations[end])
            grad_activations[end] = await model.receive(rank+1)
        else:
            grad_activations[end] = model.loss(activations[end])
        
        for l in range(end - 1, start - 1, -1):
            grad_w, grad_activations[l] = model.backward(
                l, activations[l], grad_activations[l + 1], weights[l]
            )
            if l not in grad_weights.keys():
                grad_weights[l] = grad_w
            else:
                grad_weights[l] += grad_w
            del grad_w

            # Remember to delete the activations to save memory
            if l != start:
                del grad_activations[l + 1], activations[l]
        
        if rank!=0:
            await model.send(rank-1,grad_activations[start])
            del grad_activations[start],activations[start]
        
    for l in range(start, end):
        weights[l], opt_states[l] = model.update(
            l, grad_weights[l], weights[l], opt_states[l]
        )
        model.set_final_weight(l, weights[l])
    return model

        
    



async def main():
    world_size = 4
    num_layers = 16
    global_batch_size = 2048

    dist = Dist(world_size)
    models: List[Model] = [
        Model(rank, dist, num_layers, global_batch_size) for rank in range(world_size)
    ]

    theoretical_time = get_model_theoretical_best_time(models[0])

    out = await asyncio.gather(*(part2_training_loop(model) for model in models))

    time, memory, max_mem_rank = get_training_stats(out)

    mfu = (theoretical_time / time) * 100

    print(f"MFU: {mfu}")

    write_chrome_trace(out, "./debug_traces/part2.json",get_global_batch_list(global_batch_size,global_batch_size))

    print(time, memory, max_mem_rank)


if __name__ == "__main__":
    asyncio.run(main())