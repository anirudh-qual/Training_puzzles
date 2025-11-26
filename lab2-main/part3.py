import asyncio
from typing import List

from lib import Dist, Model
from utils import (
    get_model_theoretical_best_time,
    get_training_stats,
    write_chrome_trace,
    get_global_batch_list,
)

def copy_activations(src, dest):
    for key, val in src.items():
        dest[key] = val

async def part3_training_loop(model: Model, batch_size: int) -> Model:
    """
    Implement distributed training using 1F1B  (1 forward pass, 1 backward pass) scheduling.
    The default 1F1B strategy should be implemented. See paper here : https://arxiv.org/pdf/2104.04473

    We will need to split layers across world_size. Each rank would work on a subset of layers but all the batches.

    It is guaranteed that the number of layers is divisible by the number of world_size. (i.e. num_layers % world_size == 0)
    It is also guaranteed that the number of batches is divisible by the number of world_size. (i.e. num_batches % world_size == 0)
    Coordinate the forward and backward passes across multiple GPUs in a pipelined manner. Verify the pipelined nature of the training using `visualize.ipynb`.

    Use `model.send` and `model.receive` to pass data between different world_size. See `lib.py` for more details on these functions.
    """
    
    weights, opt_states, activations, grad_activations, grad_weights = model.storage()
    rank=model.rank
    layer_per_rank = model.num_layers // model.world_size
    start = rank * layer_per_rank
    end = start + layer_per_rank

    for l in range(start, end):
        weights[l],opt_states[l]=model.load_weights(l)
    
    s_batch_id = 0
    r_batch_id = 0
    global_batch_list = get_global_batch_list(model.global_batch_size,batch_size)
    total_batches = len(global_batch_list)
    activations_hsh={}
    

    #Warmup
    for idx in range(0,model.world_size-rank):
        microbatch = global_batch_list[idx]
        if rank!=0:
            activations[start] = await model.receive(source=rank-1)
        else:
            activations[start] = model.get_input_activation(microbatch)
        
        for l in range(start, end):
                activations[l + 1] = model.forward(l, activations[l], weights[l])

        if rank != model.world_size-1:
            await model.send(rank+1,activations[end])       
        activations_hsh[idx]=activations.copy()
        

    s_batch_id += model.world_size-rank

    while s_batch_id < total_batches:
        #Backward 
        copy_activations(activations_hsh[r_batch_id],activations)
        
        if rank != model.world_size-1:
            grad_activations[end] = await model.receive(rank+1)
        else:
            grad_activations[end] = model.loss(activations[end])
        
        for l in range(end-1,start-1,-1):
            grad_w, grad_activations[l] = model.backward(
                    l, activations[l], grad_activations[l + 1], weights[l]
                )
            if l not in grad_weights.keys():
                grad_weights[l] = grad_w
            else:
                grad_weights[l] += grad_w
            del grad_w
            
            if l != start:
                    del grad_activations[l + 1], activations[l]
            
        if rank!=0:
            await model.send(rank-1,grad_activations[start])
            del grad_activations[start],activations[start]
        del activations_hsh[r_batch_id]
        r_batch_id+=1

        #Forward
        microbatch = global_batch_list[s_batch_id]
        if rank!=0:
            activations[start] = await model.receive(source=rank-1)
        else:
            activations[start] = model.get_input_activation(microbatch)
        for l in range(start, end):
                activations[l + 1] = model.forward(l, activations[l], weights[l])
                
        
        if rank != model.world_size-1:
            await model.send(rank+1,activations[end])
           
        activations_hsh[s_batch_id]=activations.copy()
        s_batch_id+=1

    while r_batch_id < total_batches:
        copy_activations(activations_hsh[r_batch_id],activations)
        if rank != model.world_size-1:
            grad_activations[end] = await model.receive(rank+1)
        else:
            grad_activations[end] = model.loss(activations[end])
        
        for l in range(end-1,start-1,-1):
            grad_w, grad_activations[l] = model.backward(
                    l, activations[l], grad_activations[l + 1], weights[l]
                )
            if l not in grad_weights.keys():
                grad_weights[l] = grad_w
            else:
                grad_weights[l] += grad_w
            del grad_w
            
            if l != start:
                    del grad_activations[l + 1], activations[l]
            
        if rank!=0:
            await model.send(rank-1,grad_activations[start])
            del grad_activations[start],activations[start]
        
        del activations_hsh[r_batch_id]
        
        r_batch_id+=1

    for l in range(start, end):
        weights[l], opt_states[l] = model.update(
            l, grad_weights[l], weights[l], opt_states[l]
        )
        model.set_final_weight(l, weights[l])
    return model


async def main():
    world_size = 4
    num_layers = 32
    global_batch_size = 2048
    batch_size = 128

    global_microbatches = []
    for i in range(global_batch_size // batch_size):
        microbatch_i = list(range(i * batch_size, (i + 1) * batch_size))
        global_microbatches.append(microbatch_i)

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

    theoretical_time = get_model_theoretical_best_time(models[0])

    out = await asyncio.gather(
        *[part3_training_loop(models[i], batch_size) for i in range(world_size)]
    )

    execution_time, max_memory, arg_max_memory = get_training_stats(out)

    mfu = (theoretical_time / execution_time) * 100
    print(f"MFU: {mfu}")

    write_chrome_trace(out, "./debug_traces/part3.json",get_global_batch_list(global_batch_size,batch_size))


if __name__ == "__main__":
    asyncio.run(main())
