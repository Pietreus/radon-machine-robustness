# Breaking Down the Radon Machine: the Geometry of a Robust Aggregation Scheme for WAFL@ECML-PKDD 2024

This repository contains an implementation of the [Radon Machine](https://proceedings.neurips.cc/paper/2017/hash/38811c5285e34e2e3319ab7d9f2cfa5b-Abstract.html) 
along with the experimental setup for the paper "Breaking Down the Radon Machine: the Geometry of a Robust Aggregation Scheme", accepted at the [2nd Workshop on Advancements in Federated Learning](https://wafl2024.di.unito.it/) at ECML-PKDD 2024.

## Rpository Structure

```
├── radon_machine/                      # Source code for the Radon-Machine model
│   ├── radon_point/                    # Source code for Radon point computations
│   └── linear_base_radon_machine.py    # Core model implementation
├── real_data_experiments/              # Experiments using the radon machine with linear base learners
├── synthetic_experiments/              # Synthetic experiments simulating the randomness of the aggregation
├── README.md                           # This README file
└── requirements.txt                    # List of dependencies

```

## Installation 
```bash
git clone https://github.com/Pietreus/radon-machine-robustness.git
cd radon-machine-robustness
pip install -r requirements.txt
```

## Datasets

We use following datasets for our experiments:
- [`SUSY`](https://doi.org/10.24432/C5460)
- [`SEA(50)`](https://www.openml.org/search?type=data&sort=version&status=any&order=asc&exact_name=SEA(50))
- [`codrna`](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-173)

As the `codrna` dataset is not easily accessible for downloads online, a preprocessed version is attached to the repository with Git LFS.
It can be fetched separately with 
```bash
git lfs pull
```

## Usage

### Running the Model

The implementation for the `RadonMachineLinearBase` class inherits from the `LinearClassifierMixin` type.
Consequently, it should behave as a linear classifier model from the scikit-learn Python library.
A short example of how to run the model is located at the bottom of the file.

### Experiments

TODO

### Simulating the Aggregation

TODO

## Results

TODO
For more details, please refer to the paper.

## Licence

TODO

## Acknowledgments

TODO
