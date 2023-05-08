# LSShadows

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://RaimelMedina.github.io/LSShadows.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://RaimelMedina.github.io/LSShadows.jl/dev/)
[![Build Status](https://github.com/RaimelMedina/LSShadows.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/RaimelMedina/LSShadows.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/RaimelMedina/LSShadows.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/RaimelMedina/LSShadows.jl)


# Locally Scrambled shadow tomography
Code implements the locally scrambled shadow tomography protocol introduced in this [`reference`](https://arxiv.org/abs/2107.04817).

This a work in progress!!

## Usage

```julia
using LSShadows
using BitBasis
using Yao
using Random
using Statistics



seed = 123

rng = MersenneTwister(seed);

N = 8
samp = 6000 # number of samples
p = 4
u = UnitaryCircuit(N, p);
idx = rand(rng, 0:2^N-1, samp);

stateEnsemble = generate_prior_povm(u, idx, rng);

computationalBasis = map(x->bitarray(x, N ÷ 2), qbasis(N ÷ 2))
oddRegSet = map(x->findall(y->y==1, bit_change_basis(x, "odd")), computationalBasis);

mean_entanglement_feature = mean(map(x->entanglement_feature_vector(x, oddRegSet), stateEnsemble));

rA = reconstruction_coefficient(mean_entanglement_feature); # get reconstruction coefficient
ψunknown = rand_state(N)


stateEnsembleMeasurement = generate_posterior_povm(Val(:RandomParams), u, ψunknown , samp, rng); #Generates the posterior POVM

estimate_fidelity(ψunknown, rA, stateEnsembleMeasurement)

```