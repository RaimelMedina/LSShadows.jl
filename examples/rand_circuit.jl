using LSShadows
using BitBasis
using Yao
using Random
using Statistics

seed = 123

rng = MersenneTwister(seed);

N = 8
samp = 5000 # number of samples
p = 5
u = UnitaryCircuit(N, p);
idx = rand(rng, 0:2^N-1, samp);

stateEnsemble = generate_prior_povm(u, idx, val=Val(:OnsiteHaar));

computationalBasis = map(x->bitarray(x, N ÷ 2), qbasis(N ÷ 2))
oddRegSet = map(x->findall(y->y==1, bit_change_basis(x, "odd")), computationalBasis);

mean_entanglement_feature = mean(map(x->entanglement_feature_vector(x, oddRegSet), stateEnsemble));

rA = reconstruction_coefficient(mean_entanglement_feature); # get reconstruction coefficient

ψunknown = rand_state(N)
stateEnsembleMeasurement = generate_posterior_povm(Val(:RandomParams), u, ψunknown , samp, val=Val(:OnsiteHaar)); #Generates the posterior POVM