# FlightEvo
## Background
The code in this repository was used for winning the [ICRA 2022 DodgeDrone Challenge](https://uzh-rpg.github.io/icra2022-dodgedrone/) in the vision-based category. It uses a [fork of Agile Flight](https://github.com/ykeuter/agile_flight) for the training environment and depends on a [fork of NEAT Python](https://github.com/ykeuter/neat-python) and a [fork of PyTorch NEAT](https://github.com/ykeuter/PyTorch-NEAT).
## Installation
- First, make sure you complete the installation of the [fork of Agile Flight](https://github.com/ykeuter/agile_flight). Only the ROS version is required.
- Next, make sure to `pip install` the [fork of NEAT Python](https://github.com/ykeuter/neat-python) and the [fork of PyTorch NEAT](https://github.com/ykeuter/PyTorch-NEAT).
- Finally, `pip install` this actual repository.
## Usage
- First launch ROS by running `roslaunch cfg/tools.launch` in a terminal.
- Next:
  - either, for training, run `python -m flightevo.dodge_trainer` in a separate terminal,
  - or, for evalutation, run `python -m flightevo.dodge_trainer` in a separate terminal.

You can have a look at the arguments of these modules to play around with different settings.
## Results
For a report on the methodology and final results, please look in the [results folder](./results/).
## Acknowledgements
Next to the contributors of the original repositories that were used, this repository is created in collaboration with [guidoAI](https://github.com/guidoAI) and [NPU-yuhang](https://github.com/NPU-yuhang), and the [MAVLab](https://github.com/tudelft) organization.
