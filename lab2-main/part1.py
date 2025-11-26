import asyncio
from typing import List

from lib import Dist, Model
from utils import (
    get_model_theoretical_best_time,
    get_training_stats,
    write_chrome_trace,
    get_next_microbatch
)


async def part1_training_loop(model: Model, batch_size: int) -> Model:
    """
    Implement distributed training using ZeRO (Zero Redundancy Optimizer) optimization.

    Shard model parameters, gradients, and optimizer states across GPUs (ranks).
    Coordinate parameter updates and gradient reductions across GPUs.
    Implement efficient memory management by only creating the full model params when needed
    """
    weights, opt_states, activations, grad_activations, grad_weights = model.storage()
    rank=model.rank
    grad_weights_sharded = [None]*model.num_layers
    weights_sharded = {}

    for l in range(model.num_layers):
        weights_sharded[l],opt_states[l]=model.load_weights(l,shard=rank,num_shards=model.world_size)
    
    for microbatch in get_next_microbatch(
        model.global_batch_size,model.world_size,batch_size,model.rank
    ):
        activations[0] = model.get_input_activation(microbatch)

        for l in range(model.num_layers):
            weights[l] = await model.all_gather(weights_sharded[l],l)
            activations[l + 1] = model.forward(l, activations[l], weights[l])
            del weights[l]
        
        grad_activations[model.num_layers] = model.loss(activations[model.num_layers])
        for l in range(model.num_layers - 1, -1, -1):
            weights[l] = await model.all_gather(weights_sharded[l],l)
            grad_weights[l], grad_activations[l] = model.backward(
                l, activations[l], grad_activations[l + 1], weights[l]
            )
            
            del grad_activations[l + 1], activations[l], weights[l]
            g_shard = await model.reduce_scatter(grad_weights[l],l)
            if grad_weights_sharded[l] == None:
                grad_weights_sharded[l]=g_shard
            else:
                grad_weights_sharded[l]+=g_shard
            del grad_weights[l]
            del g_shard
        

    for l in range(model.num_layers):
        weights[l]=weights_sharded[l]
        grad_weights[l] = grad_weights_sharded[l]
        weights[l], opt_states[l] = model.update(
            l, grad_weights[l], weights[l], opt_states[l]
        )
        model.set_final_weight(l,weights[l])
    return model

async def main():
    world_size = 8
    num_layers = 36
    global_batch_size = 32
    batch_size = 4

    dist = Dist(world_size)
    models: List[Model] = [
        Model(
            num_layers=num_layers,
            global_batch_size=global_batch_size,
            rank=i,
            dist=dist,
        )
        for i in range(world_size)
    ]

    theoretical_best_time = get_model_theoretical_best_time(models[0])

    out = await asyncio.gather(
        *[
            part1_training_loop(
                models[i],
                batch_size,
            )
            for i in range(world_size)
        ]
    )

    time, memory, max_mem_rank = get_training_stats(out)

    mfu = (theoretical_best_time / time) * 100

    print(mfu)

    write_chrome_trace(out, "./debug_traces/part1.json")


if __name__ == "__main__":
    asyncio.run(main())
