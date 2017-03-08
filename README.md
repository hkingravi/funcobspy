# Function Observers
This is a Python toolbox for code pertaining to function observers.
The prototypical example of a functional observer is the
kernel observer and controller paradigm. The primary goal of these methods is the modeling and control of
spatiotemporally varying processes (i.e. stochastic phenomena that vary
over space AND time). Practical applications of these types of methods
include ocean temperature modeling and monitoring, control of diffusive
processes in power plants, optimal decision-making in contested areas with a
patrolling enemy, disease propagation in urban population centers, and so on.
This repository will add code where the generator for the function space will
be generalized, and will include deep neural networks. 


# Setup

To get this repo, and to install all of the dependencies, run the following commands.

```bash
git clone https://github.com/hkingravi/funcobspy.git  # clone repo (https)
cd funcobspy
virtualenv funcenv  # create virtual environment with all the of dependencies required
source funcenv/bin/activate  # activate virtual environment
chmod +x install_dependencies.sh  # make install script an executable
./install_dependencies.sh  # install all reqs using pip, fix matplotlib backend issue
python setup.py develop  #  install funcobspy in a development environment
```

