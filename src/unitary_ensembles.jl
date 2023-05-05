"""
`RandomCircuit` object. The choice of single-qubit and multi-qubit gates that
conform a `layer` is passed as an argument. Here we support `Val(:XZXCnot)` 

# Arguments
* `N::Int`: Number of qubits
* `p::Int`: Circuit depth
"""
struct UnitaryCircuit
    N::Int
    p::Int
end    

##############################################
#### Defining some unitary circuit layers ####
##############################################
"""
    xzxcnot_circuit(f::UnitaryCircuit)

Create an XZXCnot circuit compossed of `Rx`-`Rz`-`Rx`-`Cnot`
gates with parameter to be dispatched. Here periodic boundary
conditions are used
"""
function xzxcnot_circuit(f::UnitaryCircuit)
    layer = chain(f.N)
    for _ ∈ f.p
        append!(layer, chain(f.N, put(loc=>Rx(0.0)) for loc = 1:f.N))
        append!(layer, chain(f.N, put(loc=>Rz(0.0)) for loc = 1:f.N))
        append!(layer, chain(f.N, put(loc=>Rx(0.0)) for loc = 1:f.N))
        for j = 1:f.N-1
            append!(layer, chain(f.N, cnot(j, j+1)))
        end
        append!(layer, chain(f.N, cnot(f.N, 1)))
    end
    
    return layer 
end

"""
    onsite_haar_circuit(f::UnitaryCircuit)

Create an OnsiteHaar circuit.
"""
function onsite_haar_circuit(f::UnitaryCircuit)
    n = f.N
    U_list = map(x -> rand(CircularEnsemble{2}(2)), collect(1:n))
    layer = foldl(kron, U_list)
    return matblock(layer, tag="⊗ⁿᵢU(2)")
end

"""
    zzx_circuit(f::UnitaryCircuit)

Creates a zzx_circuit compossed of `Rz`-`Rzz`-`Rx`
gates with parameter to be dispatched. Here periodic boundary
conditions are used
"""
function zzx_circuit(f::UnitaryCircuit)
    layer = chain(f.N)
    for _ ∈ f.p
        append!(layer, chain(f.N, put(loc=>Rz(0.0)) for loc = 1:f.N))
        for j = 1:f.N-1
            append!(layer, chain(f.N, put(f.N, (j, j+1) => rot(ZZ, 0.0))))
        end
        append!(layer, chain(f.N, put(f.N, (f.N, 1) => rot(ZZ, 0.0))))
        append!(layer, chain(f.N, put(loc=>Rx(0.0)) for loc = 1:f.N))
    end
    return layer 
end

####################################################
#### Constructing unitaries from circuit layers ####
####################################################

"""
    construct_unitary(f::UnitaryCircuit; val=Val(:XZXCnot))

Returns the unitary operator representing the `N`-qubit and depth `p`
unitary circuit with the choice of layer given by the parameter `type`

# Arguments
* `f::RandomCircuit`: `UnitaryCircuit` object

## Optional arguments
* `val`: By default equal to `Val(:XZXCnot)`. Other options are `Val(:ZZX)`
corresponding to the Floquet Ising Circuit and `Val(:OnsiteHaar)` for a local
circuit composed of tensor products of Haar-random single qubit unitaries. 
"""
function construct_unitary(f::UnitaryCircuit; val=Val(:XZXCnot))
    if val == Val(:OnsiteHaar)
        return onsite_haar_circuit(f)
    elseif val == Val(:ZZX)
        return zzx_circuit(f)
    else
        return xzxcnot_circuit(f)
    end
end


function construct_state(f::UnitaryCircuit, params::Vector{Float64}; compBasisIndex::Int64=0, val=Val(:XZXCnot))
    U = construct_unitary(f; val = val)
    if val != Val(:OnsiteHaar)
        dispatch!(U, params)
    end
    ψ = product_state(ComplexF64, f.N, compBasisIndex)
    return ψ |> U'
end

function construct_state(::Val{:RandomParams}, f::UnitaryCircuit; compBasisIndex::Int64=0, val=Val(:XZXCnot))
    U = construct_unitary(f; val = val)
    if val != Val(:OnsiteHaar)
        dispatch!(U, 2π*rand(nparameters(U)))
    end
    ψ = product_state(ComplexF64, f.N, compBasisIndex)
    return ψ |> U'
end

######################################################
#### Generating ensemble of states from repeating ####
#### the sampling following H.K.P protocol        ####
######################################################

function generate_prior_povm(f::UnitaryCircuit, compBasisIndexVector::Vector{Int64}; val=Val(:XZXCnot))
    ensemble = map(x -> construct_state(Val(:RandomParams), f; compBasisIndex=x, val=val), compBasisIndexVector)
    return ensemble
end

function simulate_measurement(::Val{:RandomParams}, f::UnitaryCircuit, ψ::ArrayReg; val=Val(:XZXCnot))
    unitary = construct_unitary(f; val=val)
    dispatch!(unitary, 2π*rand(nparameters(unitary)))

    b = copy(ψ)
    b |> unitary

    computBasisVec = (b |> r->measure(r))[1]
    
    return ArrayReg(computBasisVec) |> unitary'
end

function generate_posterior_povm(::Val{:RandomParams}, f::UnitaryCircuit, ψ::ArrayReg, samples::Int; val=Val(:XZXCnot))
    ensemble = map(x -> simulate_measurement(Val(:RandomParams), f, ψ; val=val), 1:samples)
    return ensemble
end