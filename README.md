# Scanr: Source Code for 'Attentive Head Scanning Drives One-Trial Place Field Potentiation' Project

This repository contains all of the source code used to conduct the data processing, behavioral and electrophysiological characterizations, statistics, and visualizations for the following paper:

> Monaco JD, Rao G, Roth ED, and Knierim JJ. (2014). Attentive scanning behavior drives one-trial potentiation of hippocampal place fields. *Nature Neuroscience*, **17**(5), 725–731.

**Abstract**

> The hippocampus is thought to have a critical role in episodic memory by incorporating the sensory input of an experience onto a spatial framework embodied by place cells. Although the formation and stability of place fields requires exploration, the interaction between discrete exploratory behaviors and the specific, immediate and persistent modifications of neural representations required by episodic memory has not been established. We recorded place cells in rats and found that increased neural activity during exploratory head-scanning behaviors predicted the formation and potentiation of place fields on the next pass through that location, regardless of environmental familiarity and across multiple testing days. These results strongly suggest that, during the attentive behaviors that punctuate exploration, place cell activity mediates the one-trial encoding of ongoing experiences necessary for episodic memory.

More information and links to the article online can be found [on my website](http://jdmonaco.com/scanning).

## Files

* `LICENSE.txt`: the license file governing use of this code
* `README.md`: this README file
* `conda-requirements.txt`: conda environment file for recreating the Anaconda Python environment used for the project
* `setup.py`: setuptools installation file

## Directories

Common library code is in the `scanr` subdirectory. For instance, the detection of head scanning events is handled by the `BehaviorDetector` class in the `scanr.behavior` module.

The analysis and reporting code is in the `scanning` subdirectory. For instance, the predictive value (scan-potentiation index or SPI) analyses from the paper were computed using the methods of the `PredictiveValueAnalysis` class in the `scanning.predictive` module.

* `scanr/`: common library modules used for the analysis
    * `config/`: parameter configuration handling and defaults (in `defaults.rc`)
    * `lib/`: a subpackage that aggregates all relevant imports (for `from scanr.lib import *`)
    * `tools/`: a toolbox of numerical, statistical, data, and image handling functionality
* `scanning/`: collection of data analysis modules and report generators
    * `core/`: a subpackage with base classes used for all analysis and report modules
* `scripts/`: scripts that were used to import raw datasets and preprocess the data
