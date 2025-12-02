
+This repository contains four progressively complex training loops for exploring distributed deep learning strategies. Each part builds on a common simulation API defined in `lib.py` and uses helper utilities from `utils.py` to gather microbatches, collect timing/memory metrics, and emit Chrome trace files for visualization.
+
+## Environment setup
+1. Create and activate a Python 3.8+ virtual environment.
+2. Install dependencies: `pip install -r requirements.txt`.
+3. Install the bundled `calculon` simulator: `pip install ./third_party/calculon`.
+
+## Common components
+All parts rely on the `Model` and `Dist` primitives in `lib.py`:
+- `Model.storage()` provides per-rank buffers for weights, optimizer state, activations, and gradients.
+- `Model.forward`, `Model.backward`, `Model.update`, and `Model.loss` implement computation on a single rank.
+- Communication helpers include `all_reduce`, `all_gather`, `reduce_scatter`, `send`, and `receive` for synchronizing parameters and exchanging activations/gradients.
+- Utility helpers `get_next_microbatch` and `get_global_batch_list` (in `utils.py`) enumerate microbatches for the configured world size.
+
+Chrome trace output is written to `debug_traces/partX.json` to visualize per-rank timelines in `chrome://tracing`.
+
+## Training loops by part
+### Part 0 – Data parallel baseline (`part0.py`)
+Demonstrates distributed data parallel (DDP) training with gradient accumulation. Each rank:
+- Loads its shard of weights/optimizer state.
+- Processes microbatches end-to-end (forward then backward) and deletes intermediate activations to save memory.
+- Sums gradients across ranks via `all_reduce` before a single weight update step after all microbatches finish.
+
+### Part 1 – ZeRO optimizer (`part1.py`)
+Implements ZeRO-style sharding of parameters, gradients, and optimizer state. Key behaviors:
+- Each rank loads only its parameter shard and gathers full weights on-the-fly for computation via `all_gather`.
+- Gradients are reduced into shards with `reduce_scatter`, then accumulated locally.
+- Final weight updates run per-shard before materializing final weights on each rank.
+
+### Part 2 – Pipeline parallelism (`part2.py`)
+Splits the model layers evenly across ranks and pipelines full batches:
+- Each rank owns a contiguous block of layers, forwards activations downstream with `send`, and receives upstream gradients.
+- Backward gradients are accumulated per-layer before local optimizer updates.
+- Activations for the pipeline boundaries are sent/received each microbatch to maintain the pipeline flow.
+
+### Part 3 – 1F1B pipeline schedule (`part3.py`)
+Implements the 1F1B (one-forward/one-backward) schedule for pipelined training:
+- Layers are partitioned as in Part 2, but microbatches are staggered so forward and backward passes overlap.
+- A warmup phase fills the pipeline; subsequent iterations alternate backward on earlier microbatches and forward on new ones.
+- Activations are cached per microbatch and reused during backward before being discarded to limit memory.
+
+### Part 4 – Custom strategy (`part4.py`)
+A stub for experimenting with bespoke distributed strategies. Suggested directions include interleaved 1F1B, hybrid data/pipeline parallelism, gradient accumulation, or activation checkpointing. Implementations should maximize MFU while respecting memory limits; update `part4_training_loop` to prototype your approach.
+
+## Running a part
+Each `partX.py` includes a `main` entrypoint that constructs matching `Model` instances across ranks, executes the training loop with `asyncio.gather`, reports MFU, and writes a Chrome trace. For example, run Part 1 with:
+
+```bash
+python part1.py
+```
+
+Adjust `world_size`, `num_layers`, `global_batch_size`, and per-part batch settings in the script to explore different scaling behaviors.
