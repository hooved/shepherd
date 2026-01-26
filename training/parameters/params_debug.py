# Debug config: identical model architecture, single-GPU/single-process for debugging
# Import everything from the real training config, override only training settings

from parameters.params_x1x3x4_diffusion_mosesaq_20240824 import params as base_params
from copy import deepcopy

params = deepcopy(base_params)

# Override only training settings for debugging (model architecture unchanged)
params['training'].update({
    'batch_size': 2,              # Smaller batch for faster iteration
    'accumulate_grad_batches': 1,
    'num_gpus': 1,                # Single GPU - no DDP
    'num_workers': 0,             # No multiprocessing - debugger friendly
    'multiprocessing_spawn': False,
    'log_every_n_steps': 10000,   # Rarely save checkpoints for debugging
    'output_dir': 'debug/',       # Separate dir to avoid checkpoint conflicts
})
