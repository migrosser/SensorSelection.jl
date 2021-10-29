# SensorSelection.jl

*Sequential forward selection and sequential backwards selection algorithms for solving sensor-selection problems in julia*

## Introduction
Consider a signal $\mathbf{x}_\Omega\in\mathbb{C}^N$, which is related to a set of measured data $\mathbf{y}_\Omega\in\mathbb{C}^M$ via a linear model
$$
\mathbf{y}_\Omega = \mathbf{H}_\Omega\mathbf{x}+\boldsymbol{\eta}_\Omega
$$
with zero-mean Gaussian measurement noise $\boldsymbol{\eta}_\Omega\sim \mathcal{N}(\mathbf{0},\mathbf{R}_\Omega)$ with covariance matrix $\mathbf{R}_\Omega$. Here each row of the measurement matrix $\mathbf{H}_\Omega\in\mathbb{C}^{M\times N}$ is a measurement vector from the set $\{\mathbf{h}_j^T \vert j=1,\dots,M_\Gamma\}$. Moreover, the set $\Omega \subset \{1,\dots,N\}$ contains the indices associated with the measurements performed by $\mathbf{H}_\Omega$. These types of models arise in a large variety of contexts such as imaging or the placement of physical sensors. 

The fundamental task at hand is to estimate $\mathbf{x}$ from the measurements $\mathbf{y}_\Omega$. Sensor selection aims to determine a measurement set $\Omega$ containing $M$ measurements, such that the uncertainty associated with the estimation of the parameter $\mathbf{x}$ is minimized. To do so, we look for a set of measurements minimizing a cost function of the form
$$
\underset{\Omega}{\text{argmin}} \ \text{tr}\large\{ (\mathbf{\Sigma}_{\mathbf{x}}^{-1}+\mathbf{H}_\Omega^H\mathbf{R}_\Omega^{-1}\mathbf{H}_\Omega)^{-1}\large\} \quad \text{subj. to} \quad |\Omega|=M.
$$
Here we assume a prior distribution $\mathbf{x}\sim \mathcal{N}(\mathbf{0},\boldsymbol{\Sigma}_{\mathbf{x}})$ for the parameter $\mathbf{x}$. In the special case $\boldsymbol{\Sigma}_{\mathbf{x}}^{-1}=\mathbf{0}$ the matrix expression inside the trace reduces to the estimation theoretic Cramer-Rao bound. If $\boldsymbol{\Sigma}_{\mathbf{x}}$ is finite, it corresponds to the covariance matrix (in the Bayesian sense) of the minimum mean squared error estimator.

The sensor selection problem is a non-convex integer programming problem. Thus, one often uses greedy approaches for its solution. Sequential forward selection starts with $\Omega = \empty$ and iteratively adds the measurements which lead to the largest decrease of the cost function.  Conversely, sequential backwards selection starts with $\Omega = \{1,\dots,N\}$ and iteratively removes the measurements leading to the smalles increase of the cost function.

This packags provides implementations of the greedy algorithms for the solution of the sensor selection problem described so far.

## Installation

Install SensorSelection.jl within julia using
```julia
import Pkg
Pkg.add("https://github.com/migrosser/SensorSelection.jl")
```

## Example
Consider the measurement set $\{\mathbf{h}_j^T = \mathbf{e}_j^T \vert j=1,\dots,M_\Gamma\}$ and a diagonal noise covariance matrix with variances in increasing order. Moreover, let us consider a diagonal prior covariance matrix of the form $\boldsymbol{\Sigma}_x = 10^{-2} \mathbf{1}$. For this example, sequential forward selection can be performed using the following code
```julia
using SensorSelection, LinearAlgebra, Random

N=128
# measurement matrix
Ht = collect( reshape( (1.0*I(N)),N,N,1 ) )
# prior variances
σx = 0.01*ones(N,1)
# measurement noise variances
σy = reshape( sort(rand(N).+eps()), N, 1)

# build experiment object
exp1 = Experiment(Ht, σy, σx, zeros(Bool, N))

# sequential forward selection
numMeas = rand(1:N) # target number of measurements in the experiment
wlog1 = optSamplingFW!(exp1, numMeas, 1, N)
```
Here the prior variances and the noise variances are specified in the form of matrices. The second dimension of these matrices can be used to describe multiple sets of variances associated to different "training examples".

Analogously to the above example, the following code performes sequential backwards selection
```julia
# same setup as before

# build experiment object
exp2 = Experiment(Ht, σy, σx, ones(Bool, N))

# sequential forward selection
wlog2 = optSamplingBW!(exp2, numMeas, 1, N)
```