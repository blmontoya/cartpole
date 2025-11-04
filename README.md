<!-- GETTING STARTED -->
# Getting Started

## Prerequisites

This program runs on a pixi environment using Python 3.11.0.

* To enter the environment, make sure to run:

    ```sh
    pixi shell -e test
    ```

All required packages are preinstalled onto the pixi environment.

Also, please run the following so you don't have to write "python" at the start of each command:

    chmod +x *.py
    

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

<!-- Using the Programs -->
# Using the Programs

## cartpole_model.py
You can create a model trained on cartpole by running:

    ```sh
    ./cartpole_model.py /PATH/<name>.safetensors
    ```

For example:

    ```sh
    ./cartpole_model.py /workspace/min_ppo.safetensors
    ```

### IMPORTANT: In ActorCritic(), record self.shared, self.actor, and self.critic. When you run your safetensors file in cartpole_eval.py. These MUST align with what you trained on the safetensors file. For convienience, I've provided two safetensor examples for what this should look like:

/workspace/min_ppo.safetensors

        ```sh
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LeakyReLU(0.01),
        )
        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)
        ```
/workspace/fast_ppo.safetensors

        ```sh
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64), nn.LeakyReLU(0.01), nn.Linear(64, 64), nn.LeakyReLU(0.01)
        )
        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)
        ```

## cartpole_eval.py
You can simulate your trained cartpole models by running:

    ```sh
    ./cartpole_eval.py <MODEL PATH> --episodes <episode count> --no-render 
    ```

By default, episodes are set to 5 and rendering is turned on.

Examples:
-> Default settings

    ```sh
    ./cartpole_eval.py /workspace/fast_ppo_cartpole.safetensors --episodes 6
    ```

-> 3 Episodes, Rendering

    ```sh
    ./cartpole_eval.py /workspace/fast_ppo_cartpole.safetensors --episodes 6
    ```

-> 20 Episodes, No rendering

    ```sh
    ./cartpole_eval.py /workspace/fast_ppo_cartpole.safetensors --episodes 20 --no-render 
    ```