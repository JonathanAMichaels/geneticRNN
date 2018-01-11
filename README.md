# geneticRNN: A simple genetic training algorithm for recurrent neural networks

**Authors:** [Jonathan A. Michaels](http://www.jmichaels.me/)

**Version:** 1.0

**Date:** 10.01.2018

## What is geneticRNN?

The current package is a Matlab implementation of a simple genetic training algorithm for recurrent neural networks. My algorithm is a very faithful implemetation of the algorithm layed out in this paper [Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning](https://arxiv.org/abs/1712.06567) as Algorithm 1.

### Abstract of Such et al. (2018):

Deep artificial neural networks (DNNs) are typically trained via gradient-based learning algorithms, namely backpropagation. Evolution strategies (ES) can rival backprop-based algorithms such as Q-learning and policy gradients on challenging deep reinforcement learning (RL) problems. However, ES can be considered a gradient-based algorithm because it performs stochastic gradient descent via an operation similar to a finite-difference approximation of the gradient. That raises the question of whether non-gradient-based evolutionary algorithms can work at DNN scales. Here we demonstrate they can: we evolve the weights of a DNN with a simple, gradient-free, population-based genetic algorithm (GA) and it performs well on hard deep RL problems, including Atari and humanoid locomotion. The Deep GA successfully evolves networks with over four million free parameters, the largest neural networks ever evolved with a traditional evolutionary algorithm. These results (1) expand our sense of the scale at which GAs can operate, (2) suggest intriguingly that in some cases following the gradient is not the best choice for optimizing performance, and (3) make immediately available the multitude of techniques that have been developed in the neuroevolution community to improve performance on RL problems. To demonstrate the latter, we show that combining DNNs with novelty search, which was designed to encourage exploration on tasks with deceptive or sparse reward functions, can solve a high-dimensional problem on which reward-maximizing algorithms (e.g. DQN, A3C, ES, and the GA) fail. Additionally, the Deep GA parallelizes better than ES, A3C, and DQN, and enables a state-of-the-art compact encoding technique that can represent million-parameter DNNs in thousands of bytes.

### Notes:

- I endeavored to make the package as flexible as possible, and therefore allows the user to pass many custom functions, including policy initialization, fitness, the physical plant, and plotting.

- I have implemented the data compression technique outlined in the original paper. In brief, policies are not passed around to and from parallel workers or stored, but rather generated from a sequence of random number seeds. While this method slows down as a function of generations, it is overall much faster and more memory efficient than other possible implementations.

- I added two regularizations that were not present in the original implementation and are optional:
    - Policies are multiplied with a decay term to prevent variance explosion as a consequence of summing many normal distributions.
    - A decay term is subtracted from the policy to bring unneeded weights closer to zero. The general effect is to produce a power law distribution of weights as opposed to normal.

- Mutation power decays automatically over generations and decays rapidly when no improved policy is found for a given generation.


## Documentation & Examples
All functions are documented throughout, and two examples illustrating the intended use of the package are provided with the release.

### Example: a delayed nonmatch-to-sample task

In the delayed nonmatch-to-sample task the network receives two temporally separated inputs. Each input lasts 200ms and there is a 200ms gap between them. The goal of the task is to respond with one value if the inputs were identical, and a different value if they were not. This response must be independent of the order of the signals and therefore requires the network to remember the first input!

related file: geneticRNN_Example_DNMS.m

### Example: a center-out reaching task

In the center-out reaching task the network needs to produce the joint angle velocities of a two-segment arm to reach to a number of peripheral targets spaced along a circle in the 2D plane, based on the desired target specified by the input.

related file: geneticRNN_Example_CO.m

## Installation Instructions

The code package runs in Matlab, and should be compatible with any version.
To install the package, simply add all folders and subfolders to the Matlab path using the set path option.

### Dependencies

The geneticRNN repository has no dependencies beyond built-in Matlab functions.