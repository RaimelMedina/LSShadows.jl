"""
    entanglement_feature(ψ::ArrayReg, r)

Calculate the entanglement feature for a given quantum state `ψ` and the register `r`.
If the length of `r` is 0 or equal to the number of qubits in `ψ`, return 1.0.
Otherwise, compute the purity of the density matrix for the specified register.
"""
function entanglement_feature(ψ::ArrayReg, r)
    if length(r) == 0 || length(r) == nqubits(ψ)
        return 1.0
    else
        dm = density_matrix(ψ, r)
        return purity(dm)
    end
end

"""
    entanglement_feature_vector(ψ::ArrayReg, reg::Vector{Vector{Int64}})

Compute the entanglement feature vector for a given quantum state `ψ` and
a vector of registers `reg`. The entanglement feature vector is a vector
containing the entanglement features for each register in `reg`.
"""
function entanglement_feature_vector(ψ::ArrayReg, reg::Vector{Vector{Int64}})
    n = nqubits(ψ)
    vEntFeature = zeros(2^(n ÷ 2))
    for i in 1:2^((n ÷ 2)-1)
        vEntFeature[i] = entanglement_feature(ψ, reg[i])
        vEntFeature[end-i+1] = vEntFeature[i]
    end
    return vEntFeature
end

"""
    entanglement_feature_page(n::Int, rA::Vector{Int64})

Calculate the entanglement feature associated with an `n`-qubit Haar random state. 
`rA` corresponds to the qubits to be kept. 
"""
function entanglement_feature_page(n::Int, rA::Vector{Int64})
    nA = length(rA)
    return (2^nA + 2^(n-nA))/(1+2^n)
end

"""
    purity(rho::DensityMatrix)

Calculate the purity of a given density matrix `rho`. The purity is defined as
the trace of the squared density matrix.
"""
function purity(rho::DensityMatrix)
    rA = size(rho |> state)[1]
    purity = 0
    for i in 1:rA
        for j in 1:rA
            purity += real(state(rho)[i, j]*state(rho)[j,i])
        end
    end
    return purity
end