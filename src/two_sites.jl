function bit_change_basis(v::BitVector, type_reg::String)
    n = 2 * length(v)
    new_bit_str = BitVector(falses(n))
    swap_places = findall(==(1), v)

    if type_reg == "odd"
        for s in swap_places
            new_bit_str[2 * s - 1] = 1
            new_bit_str[2 * s] = 1
        end
    elseif type_reg == "even"
        for s in swap_places
            new_bit_str[2 * s] = 1
            new_bit_str[mod(2 * s + 1, n)] = 1
        end
    else
        error("Only 'even' and 'odd' bit strings are possible")
    end
    
    return new_bit_str
end

qbasis(n::Int) = 0:(2^n-1)

function J(;d=2) 
    Jtensor = zeros(Float64, 2, 2, 2)
    Jtensor[1, 1, 1] = 1
    Jtensor[2, 2, 2] = 1
    Jtensor[1, 2, 2] = d/(d^2 + 1)
    Jtensor[2, 1, 1] = d/(d^2 + 1)
    Jtensor[1, 1, 2] = d/(d^2 + 1)
    Jtensor[2, 2, 1] = d/(d^2 + 1)
    return Jtensor
end

Wg(d::Int) = [[1/(d^2-1) -1/(d*(d^2-1))]; [-1/(d*(d^2-1)) 1/(d^2-1)]]


const WG = Wg(4)
const J2 = J(;d=2)

function JWg()
    Atensor = zeros(2, 2, 2, 2)
    #localHilb = [0,1]
    for s2 in 0:1
        for r1 in 0:1
            for s3 in 0:1
                for s1 in 0:1
                    Atensor[s1+1, s3+1, r1+1, s2+1] = WG[r1+1, s1+1]*J2[s1+1, s2+1, s3+1]
                end
            end
        end
    end
    return Atensor
end

const JWG = JWg();

function WgAB(rodd::Int, seven::Int, n::Int)
    transferMatrix = Matrix{Float64}(I(2))
    for i in 0:n-1
        transferMatrix *= JWG[:, :, (rodd >> i) & UInt(1) + 1, (seven >> i) & UInt(1) + 1]
    end
    return tr(transferMatrix)
end

# function fusionCoefficient(q::Int, z::Int, ro::Int, bitBasis::UnitRange{Int64}; d=2)
#     # bitBasis is supposed to be 0:2^(n/2)-1
#     # d is the local Hilbert space dimension which we take by default to be 2 (spin degree of freedom)
#     # It is then clear that bitBasis[end] + 1 = 2^(n/2) => log2(bitBasis[end] + 1) = log2(2^(n/2)) = n/2
#     n = 2Int(log2(bitBasis[end]+1))

#     fusionC, numSpins_se = 0.0, 0.0
#     for se in bitBasis
#         if z == q & se
#             numSpins_se = 2*count_ones(se)#sum(bitarray(se,n ÷ 2))
#             fusionC += d^(2n-numSpins_se)*WgAB(ro, se, n ÷ 2)
#         end
#     end
#     return fusionC
# end

function fusion_coefficient(q::Int, z::Int, ro::Int, bit_basis::UnitRange{Int64}; d=2)
    # bit_basis is supposed to be 0:2^(n/2)-1
    # d is the local Hilbert space dimension, 
    # which we take by default to be 2 (spin degree of freedom)
    n = 2 * Int(log2(bit_basis[end] + 1))
    d_power_2n = d^(2 * n)

    fusion_c = 0.0
    for se in bit_basis
        if z == (q & se)
            num_spins_se = 2 * count_ones(se)
            fusion_c += d_power_2n / d^num_spins_se * WgAB(ro, se, n ÷ 2)
        end
    end
    return fusion_c
end

@doc raw"""
    reconstruction_coefficient(WEσ::Vector{Float64}; d=2)

Calculate the reconstruction coefficient for a given vector of entanglement
features `WEσ`. 

# Arguments
- `WEσ`: Vector of entanglement features.
- `d`: Optional argument with default value of 2, representing the dimension of the underlying quantum system.

# Returns
- Reconstruction coefficient as a vector of Float64.
"""
function reconstruction_coefficient(WEσ::Vector{Float64}; d=2)
    n = 2 * Int(log2(length(WEσ)))
    basis_n_half = qbasis(n ÷ 2)
    M = UpperTriangular(zeros(2^(n ÷ 2), 2^(n ÷ 2)))
    
    Threads.@threads for q in basis_n_half
        for z in 0:q
            for x in basis_n_half
                M[z + 1, q + 1] += fusion_coefficient(q, z, x, basis_n_half; d=d) * WEσ[x + 1]
            end
        end
    end

    b = zeros(2^(n ÷ 2))
    b[end] = 1.0

    return M \ b
end

@doc raw"""
    partial_tr_embedding!(partial_tr::Array{ComplexF64, 2}, ψ0::ArrayReg, reg::Vector{Int64})

Compute the reduced density matrix ``\rho_A`` of a given quantum state `ψ0` in the region `reg`. After that, it
embeds the reduced density matrix back into the full hilbert space by constructing 

```math 
\rho_A \otimes \mathbb{I}/d_{\bar{A}}
```
The results are stored in a pre-allocated array `partial_tr`.

# Arguments
- `partial_tr`: Pre-allocated matrix to store the result.
- `ψ0`: Quantum state as an ArrayReg.
- `reg`: Vector of Int64 representing the qubits to keep.

# Returns
- None. The function updates the `partial_tr` matrix in-place.
"""
function partial_tr_embedding!(partial_tr::Array{ComplexF64, 2}, ψ0::ArrayReg, reg::Vector{Int64})
    n = nqubits(ψ0)
    if length(reg) == 0
        partial_tr .= mat(igate(n)) / (2^n)
    elseif length(reg) == n
        partial_tr .= state(ψ0) * state(ψ0)'
    else
        nB = n - length(reg) # spins on the complement of region reg
        rhoA = density_matrix(ψ0, reg);
        partial_tr .= mat(put(n, Tuple(reg) => matblock(rhoA.state))) / (2^nB)
    end
    @assert tr(partial_tr) |> real ≈ 1.0
end


@doc raw"""
    classical_snapshot!(M_inverse::Array{ComplexF64, 2}, partial_tr::Array{ComplexF64, 2}, reconst_coefficient::Vector{Float64}, ρ_unknown::ArrayReg; threaded=false)

Construct the classical snapshot of an unknown quantum state `ρ_unknown`. For this, given the reconstruction coefficient it
returns ``\sum_A r_A \sigma_A `` where `r_A` is the ``A``-th component of the reconstruction coefficient and 
``\sigma_A`` is the embeded reduced density matrix of the state ``\sigma`` on region ``A``.  

# Arguments
- `M_inverse`: Pre-allocated matrix to store the result.
- `partial_tr`: Pre-allocated matrix to store the partial trace embedding.
- `reconst_coefficient`: Vector of Float64 representing the reconstruction coefficients.
- `ρ_unknown`: Unknown quantum state as an `ArrayReg`.
- `threaded`: Optional argument with default value of `false`, which determines if the function should be run in a multi-threaded mode.

# Returns
- None. The function updates the `M_inverse` matrix in-place.
"""
function classical_snapshot!(M_inverse::Array{ComplexF64, 2}, partial_tr::Array{ComplexF64, 2}, reconst_coefficient::Vector{Float64}, ρ_unknown::ArrayReg; threaded=false)
    n = nqubits(ρ_unknown)
    computational_basis = map(x -> bitarray(x, n ÷ 2), qbasis(n ÷ 2))

    even_reg_set = map(x -> findall(==(1), bit_change_basis(x, "even")), computational_basis)

     # Reset M_inverse
     M_inverse .= ComplexF64(0)

    if threaded
        # Create a lock to synchronize access to M_inverse
        M_inverse_lock = ReentrantLock()

        # Use mapreduce with multi-threading
        M_inverse .= @sync mapreduce(i -> begin
            local_partial_tr = deepcopy(partial_tr)  # Create a local copy of partial_tr for each thread
            partial_tr_embedding!(local_partial_tr, ρ_unknown, even_reg_set[i])
            result = reconst_coefficient[i] .* local_partial_tr
            lock(M_inverse_lock) do
                M_inverse .+= result
            end
            result
        end, +, eachindex(even_reg_set), init=zeros(ComplexF64, size(M_inverse)))
    else
        M_inverse .= mapreduce(i -> begin
            partial_tr_embedding!(partial_tr, ρ_unknown, even_reg_set[i])
            reconst_coefficient[i] .* partial_tr
        end, +, eachindex(even_reg_set), init=M_inverse)
    end
end

"""
    estimate_fidelity(rho::ArrayReg, rA::Vector{Float64}, ensemble::Vector; threaded=false)

Estimate the fidelity between a given quantum state `rho` and an the reconstructed state.

# Arguments
- `rho`: Quantum state as an ArrayReg.
- `rA`: Vector of Float64 representing the reconstruction coefficients.
- `ensemble`: Vector of quantum states.
- `threaded`: Optional argument with default value of `false`, which determines if the function should be run in a multi-threaded mode.

# Returns
- Fidelity estimate as a Float64.
"""
function estimate_fidelity(rho::ArrayReg, rA::Vector{Float64}, ensemble::Vector; threaded=false)
    n = nqubits(rho)
    samples = length(ensemble)
    
    partial_tr = zeros(ComplexF64, 2^n, 2^n)
    M_inverse = similar(partial_tr)
    
    fid = mapreduce(i -> begin
        classical_snapshot!(M_inverse, partial_tr, rA, ensemble[i], threaded=threaded)
        (state(rho)' * M_inverse * state(rho))[1] |> real
    end, +, 1:samples, init=0.0)
    
    return sqrt(fid / samples)
end


# function estimate_fidelity(rho::ArrayReg, rA::Vector{Float64}, ensemble::Vector; threaded=false)
#     n = nqubits(rho)
#     samples = length(ensemble)
    
#     partial_tr = zeros(ComplexF64, 2^n, 2^n)
#     M_inverse = similar(partial_tr)
    
#     fid = mapreduce(i -> begin
#         classical_snapshot!(M_inverse, partial_tr, rA, ensemble[i], threaded=threaded)
#         (state(rho)' * M_inverse * state(rho))[1] |> real
#     end, +, 1:samples, init=0.0)
    
#     return sqrt(fid / samples)
# end

# function estimate_fidelity(rho::ArrayReg, rA::Vector{Float64}, ensemble::Vector)
#     fid = Float64(0)
#     n = nqubits(rho)
#     samples = length(ensemble)
    
#     ρClassicalShadow = zeros(ComplexF64, 2^n, 2^n)

#     for i in 1:samples
#         ρClassicalShadow .= classicalSnapshot(rA, ensemble[i])
#         fid += (state(rho)' * ρClassicalShadow * state(rho))[1] |> real
#     end
#     return sqrt(fid/samples)
# end