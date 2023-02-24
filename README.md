# README #

This repository contains a Python3 implementation of (Non-Linear) AR-HMM.

## References ##
This code is based on the following article:
https://arxiv.org/abs/2302.11834

If you use this code, please cite it.

## Installation ##
run (may require superuser permissions)

` pip3 install . `

## Usage ##
The package allows to define a model as a class.  
Each model is characterized by the number of hidden modes and the dynamics/observation space definition.
Depending on the particular model, the latter parameters is defined in different ways.
In particular

  * for GRBF\_ARHMM, Linear\_ARHMM, Quadratic\_ARHMM, and Cubic\_ARHMM the dimensionality of the space has to be defined;
  * for Unit\_Quaternion\_ARHMM, Hand\_Gripper\_ARHMM, Decoupled\_Linear\_ARHMM, Orientation\_Gripper\_ARHMM, Pose\_ARHMM, and Pose\_Gripper\_ARHMM the number of end effectors has to be defined;
  * for Generic\_Decoupled\_Linear\_ARHMM the list of decoupled dimensions has to be specified.

### class methods ###
The ARHMM object has the following methods:

  * initialize to initialize the guesses for the EM algorithm using clustering
  * em\_algorithm to perform the EM algorithm
  * viterbi to use the Viterbi algorithm

**NOTE**: EM algorithm and initialize nedd a _list_ of trajectories. If your data is a single trajectory, run the methods on a list with that trajectory as argument.

## Repository Content ##

Folder `nl\_arhmm` contains the code, while folde `demos/` contains some examples.  
`nl\_arhmm/arhmm.py` and `nl\_arhmm/dynamics.py` contain a template to create your personal ARHMM with custom dynamics.

## some remarks ###

## Contact ##

  * michele.ginesi@univr.it
  * michele.ginesi@gmail.com
