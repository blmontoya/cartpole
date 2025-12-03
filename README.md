<!-- GETTING STARTED -->
# Getting Started

## Prerequisites

This program runs on a pixi environment using Python 3.11.0.

* To enter the environment, make sure to run:

    ```sh
    pixi shell -e test
    ```

All required packages are preinstalled onto the pixi environment.

* Also, please enter the src directory and run the following command to avoid writing "python" at the start of each command:

    ```sh
    chmod +x *.py
    ```

## Optional (But Recommended)

I used TensorBoard to track learning rate, accuracy, and loss. If you would like to also track these variables and activate TensorBoard, open a new terminal and paste the line below. Afterwards, go to http://localhost:6006/.

* Running TensorBoard:

    ```sh
    tensorboard --logdir=runs
    ```

For running on a GPU using CUDA, make sure the PyTorch version with CUDA is installed.

* Running on GPU via CUDA:

    ```sh
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ```
If this does not work, please consult: https://github.com/garylvov/dev_env/tree/main/setup_scripts/nvidia

<!-- Using the Programs -->
# Using the Programs

All commands here should be run in the src directory.

## cartpole_model.py
You can create a model trained on cartpole by running:

    ./cartpole_model.py /PATH/<name>.safetensors

For example:

    ./cartpole_model.py /workspace/models/min_ppo.safetensors

### IMPORTANT: 
In ActorCritic(), record self.shared, self.actor, and self.critic. When you run your safetensors file in cartpole_eval.py. These MUST align with what you trained on the safetensors file. For convenience, I've provided two safetensor examples for what this should look like:

* /workspace/min_ppo.safetensors
    ```
    self.shared = nn.Sequential(
        nn.Linear(state_dim, 64),
        nn.LeakyReLU(0.01),
    )
    self.actor = nn.Linear(64, n_actions)
    self.critic = nn.Linear(64, 1)
    ```

* /workspace/fast_ppo.safetensors
    ```
    self.shared = nn.Sequential(
        nn.Linear(state_dim, 64), nn.LeakyReLU(0.01), nn.Linear(64, 64), nn.LeakyReLU(0.01)
    )
    self.actor = nn.Linear(64, n_actions)
    self.critic = nn.Linear(64, 1)
    ```

## cartpole_eval.py
You can simulate your trained cartpole models by running:

    ./cartpole_eval.py <MODEL PATH> --episodes <episode count> --no-render 

By default, episodes are set to 5 and rendering is turned on.

Examples:
* Default settings
    ```
    ./cartpole_eval.py /workspace/models/fast_ppo_cartpole.safetensors
    ```

* 3 Episodes, Rendering
    ```
    ./cartpole_eval.py /workspace/models/fast_ppo_cartpole.safetensors --episodes 3
    ```

* 20 Episodes, No rendering
    ```
    ./cartpole_eval.py /workspace/models/fast_ppo_cartpole.safetensors --episodes 20 --no-render 
    ```

## svd.analysis.py
You can run SVD analysis on the trained cartpole model using:

    ./svd_analysis.py <MODEL PATH>

The k-value represents the percentage of weights being used in the model.

## weight_watcher.py
The cartpole model is smaller than weight watcher typically runs on, but you can run a weight watcher program for the layers by calling:

    ./weight_watcher.py <MODEL PATH>

## lunar_walker_mlp.py
Similar to cartpole, you can train an MLP to complete both lunar lander and bipedal walker. The MLP uses the same backbone to train both lunar lander and bipedal walker, and can be run by calling:

    ./lunar_walker_mlp.py /PATH/<name>.safetensors --cycles <cycle count>

For example (The current multitask_model.safetensors):

    ./lunar_walker_mlp.py /workspace/models/multitask_model.safetensors --cycles 250

By default, cycles is set to 300.

### IMPORTANT: 
Similar to cartpole, please keep in mind that if you change anything in MultiTaskActorCritic, to make sure that it aligns with what you trained on the safetensors file.

For convenience, I've provided one safetensor example for what this should look like:

* /workspace/multitask_model.safetensors
    ```
    self.shared_backbone = nn.Sequential(
        nn.Linear(128, 768),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(768, 768),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(768, 128),
        nn.LeakyReLU(negative_slope=0.01),
    )
    # Input adapters
    self.input_adapters = nn.ModuleDict({
        "lunar": nn.Linear(8, 128),
        "walker": nn.Linear(24, 128)
    })
    # Actor heads
    self.actor_heads = nn.ModuleDict({
        "lunar": nn.Sequential(
            nn.Linear(128, 128), 
            nn.LeakyReLU(0.01),
            nn.Linear(128, 4)
        ),
        "walker": nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 6)
        )
    })
    # Critic heads
    self.critic_heads = nn.ModuleDict({
        "lunar": nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1)
        ),
        "walker": nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1)
        )
    })
    self.log_std = nn.Parameter(torch.zeros(6))
    self.current_task = "lunar"
    ```

 ## lunar_walker_eval.py
 You can simulate your trained lunar/walker models by running:

    ./lunar_walker_eval.py <MODEL PATH> --task <task type> --episodes <episode count> --no-render --stochastic

By default, the task is Lunar Lander, episodes are set to 5, rendering is enabled, and the mode is deterministic (best action).

Examples:
* Default settings
    ```
    ./lunar_walker_eval.py /workspace/models/multitask_model.safetensors
    ```

* Bipedal Walker, 3 Episodes, Rendering
    ```
    ./lunar_walker_eval.py /workspace/models/multitask_model.safetensors --task walker --episodes 3
    ```

* Lunar Lander, Default Episodes, No Rendering, Stochastic (random sampling)
    ```
    ./lunar_walker_eval.py /workspace/models/multitask_model.safetensors --task lunar --no-render --stochastic
    ```
    
