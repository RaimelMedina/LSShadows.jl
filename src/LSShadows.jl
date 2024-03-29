module LSShadows

export qbasis, bit_change_basis, WgAB, fusion_coefficient, reconstruction_coefficient, partial_tr_embedding!, classical_snapshot!

export UnitaryCircuit, xzxcnot_circuit, zzx_circuit, onsite_haar_circuit, construct_unitary, construct_state, generate_prior_povm, generate_posterior_povm
export entanglement_feature, entanglement_feature_page, entanglement_feature_vector, purity

export estimate_fidelity

using Yao
using BitBasis
using Base.Threads
using Distributions
using Combinatorics
using Statistics
using Random
using YaoPlots
using SparseArrays
using LinearAlgebra
using QuantumInformation

include("unitary_ensembles.jl")

include("entanglement_feature.jl")

include("two_sites.jl")

@const_gate ZZ = mat(kron(Z,Z))
@const_gate YY = mat(kron(Y,Y))
@const_gate XX = mat(kron(X,X))

export ZZ, YY, XX

end