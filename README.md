# LinearRL - Python Implementation
This repository is the code accompanying ["Linear reinforcement learning in planning, grid fields, and cognitive control"](https://www.nature.com/articles/s41467-021-25123-3).

## Introduction
Everything should be self-contained inside of this repo. If you have any troubles running the code or if you have any quesstions you can reach out to me via email or make a GitHub issue.

## Usage
### Conda Environment
I recommend creating a conda environment for usage with this repo. Especially because we will be installing some custom built environments. You can install the conda environment from the yml file I have provided.
```bash
conda env create -f env.yml
```

### RL Environments
Because we are using custom gym environments, you need to install them locally in order for gymansium to recognize them when we are constructing our environment variables. To install the environemnts, just run:
```bash
pip install -e gym-env
```

### Run
To run the LinearRL model you can run `main.py` which will use LinearRL to solve for the optimal value function and optimal policy on the specified environment.
```bash
python src/main.py
```

## Notebooks
Each notebook contains the simulations to generate the figures in the paper all the notebooks can be found in `src/`.
* Figure 1 - `occupancy_map.ipynb`
* Figure 2 - 2d. `convergence.ipynb`, 2e/f. `complex_maze.ipynb`
* Figure 3 - 3c/d. `latent.ipynb`, 3g/h. `detour.ipynb`
* Figure 4 - `de-cothi-analysis/`
* Figure 5 - `reval_nhb.ipynb`
* Figure 6 - `reval_nhb.ipynb`
* Figure S1 - `de-cothi-analysis/`
* Figure S2 - `reval_maze.ipynb`