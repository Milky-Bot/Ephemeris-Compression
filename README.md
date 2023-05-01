# Ephemeris-Compression
GPS Ephemeris compression using an Extreme Learning Machine Regression. We solve the problem of finding phases and frequencies with stochasticity.

The GPS ephemeris contains information about the location and velocity of each satellite in the GPS constellation, as well as other parameters that are necessary for accurate positioning on the ground. As there are bandwidth limitations when transmitting GPS ephemeris data from the space vehicle to ground stations on Earth, in order to make the most efficient use of this limited bandwidth, this data is compressed before transmission using a technique called variable-length coding.
The goal of this project was to explore the possibility of using Machine Learning and Artificial intelligence algorithms for GPS ephemeris compression.
We recasted the Ephemeris Compression task as a regression problem, where the input is the time, and the output is the 3D position of a space vehicle. Employing an Extreme Learning Machine, we modelled the position as a series of sine functions of time. In particular, we chose to set frequencies and phases as a sequence of random numbers, obtaining better approximation performance than current methods, while also reducing the size of the compressed representation. This was possible thanks to the fact that the sequences obtained from random number generators are fully determined by the initial random seed.

1) ELM.ipynb does an unweighted regression for different numbers of neurons
2) weighted_elm.ipynb makes it weighted
3) Reconstruct_ephemeris.ipynb given the 2 seeds and the amplitudes, it reconstructs the ephemeris

The weighted_elm.py script encloses the ELM class with sine activation function.
