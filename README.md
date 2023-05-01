# Ephemeris-Compression
GPS Ephemeris compression using an Extreme Learning Machine Regression. We solve the problem of finding phases and frequencies with stochasticity

1) ELM.ipynb does an unweighted regression for different numbers of neurons
2) weighted_elm.ipynb makes it weighted
3) Reconstruct_ephemeris.ipynb given the 2 seeds and the amplitudes, it reconstructs the ephemeris

The weighted_elm.py script encloses the ELM class with sine activation function.
