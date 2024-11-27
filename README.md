# LinearRL - Python Implementation
This repository is the code accompanying ["Armin's amazing future publication"](https://scholar.google.com/citations?user=CvlkhhwAAAAJ&hl=en).

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

### Code synposis
The different models I tested can be found in `src/models.py`, the main model of interest for most readers is probably are `SR_IS` and `SR_IS_NHB`models. Note that there are two versions because one is defined to work on the tabular environments I constructed in `gym-env` while the other is designed to work on tree like environments that were used in ["Momennejad et al."](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=OFdUAJwAAAAJ&citation_for_view=OFdUAJwAAAAJ:Tyk-4Ss8FVUC).

I have different notebooks, each containing different simulations that I used to construct my figures. They should be relatively easy to parse through but you can also look through the next section to determine which notebook you would like to run.

## Notebooks
Each notebook contains the simulations to generate the figures in the paper all the notebooks can be found in `src/`.
* Figure 1 - `occupancy_map.ipynb`
* Figure 2 - 2d. `convergence.ipynb`, 2e/f. `complex_maze.ipynb`
* Figure 3 - 3c/d. `latent.ipynb`, 3g/h. `detour.ipynb`
* Figure 4 - `de-cothi-analysis/`
* Figure 5 - `reval_nhb.ipynb`
* Figure S1 - `de-cothi-analysis/`
* Figure S2 - `reval_maze.ipynb`
* Figure S3 - `reval_nhb.ipynb`