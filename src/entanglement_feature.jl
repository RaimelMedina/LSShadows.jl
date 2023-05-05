function entanglement_feature(ψ::ArrayReg, r)
    if length(r) == 0 || length(r) == nqubits(ψ)
        return 1.0
    else
        dm = density_matrix(ψ, r)
        return purity(dm)
    end
end
function entanglement_feature_vector(ψ::ArrayReg, reg::Vector{Vector{Int64}})
    n = nqubits(ψ)
    vEntFeature = zeros(2^(n ÷ 2))
    for i in 1:2^((n ÷ 2)-1)
        vEntFeature[i] = entanglement_feature(ψ, reg[i])
        vEntFeature[end-i+1] = vEntFeature[i]
    end
    return vEntFeature
end

function entanglement_feature_page(n::Int, rA::Vector{Int64})
    nA = length(rA)
    return (2^nA + 2^(n-nA))/(1+2^n)
end

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