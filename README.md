# m-top-covid

This repository contains code for the paper *Evaluating COVID-19 vaccine allocation policies
using Bayesian m-top exploration*

## Contents

* **envs**: The environments used for running the algorithms
* **mab**: Multi-armed bandits
* **loggers**: Logging for bandits and top-*m* algorithms
* **resources**: Vaccine supply based on weekly deliveries


## STRIDE experiments

The use of the STRIDE environment requires the vaccine extension and pybind11 wrapper implemented in the [STRIDE fork](https://github.com/icimpean/stride/tree/vaccine).
Additionally, the [synthetic population data for Belgium](https://doi.org/10.5281/zenodo.4485995) is necessary to simulate the entire 11 million population.

The full mapping of arm numbers to vaccine strategies can be found [here](Vaccine_Strategies.pdf).


## How to run experiments

Run simulations for a single arm on STRIDE
```
python mab/play_arm.py --help
```

Run simulations for a bandit on STRIDE
```
python mab/play_bandit.py --help
```

Run simulations for a bandit on the ground truth
```
python mab/play_bandit_gt.py --help
```
