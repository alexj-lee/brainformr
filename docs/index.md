# Repository quickstart

This is documentation for code written as part of the manuscript ["Data-driven fine-grained region discovery in the mouse brain with transformers"](https://www.biorxiv.org/content/10.1101/2024.05.05.592608v1). 

## Installation

* `pip install git+github.com:alexj-lee/brainformr.git` or clone and pip install.
* you will need a GPU to use this package; almost all analyses in the paper were trained on a workstation with 2 A6000 GPUs
    * without a GPU I highly recommend using a service like modal or coiled to facilitate training for a small cost

## Usage
See the [usage](usage.md) page for more details and a specific example of how to train on the Allen Institute for Brain Science mouse.

::: brainformr.data.loader_pandas

