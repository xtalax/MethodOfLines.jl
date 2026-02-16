"""
    MOLPDEProblem{P, M}

Wrapper around a SciML problem (ODEProblem/NonlinearProblem) that carries MOLMetadata.
When `solve` is called on this wrapper, it solves the inner problem and wraps the
result into a PDESolution using the stored metadata.
"""
struct MOLPDEProblem{P, M <: MOLMetadata}
    prob::P
    metadata::M
end

# Forward property access to inner problem for transparent usage
function Base.getproperty(mp::MOLPDEProblem, s::Symbol)
    if s === :prob || s === :metadata
        return getfield(mp, s)
    else
        return getproperty(getfield(mp, :prob), s)
    end
end

function Base.propertynames(mp::MOLPDEProblem, private::Bool=false)
    return (:prob, :metadata, propertynames(getfield(mp, :prob), private)...)
end

function DiffEqBase.solve(mp::MOLPDEProblem, args...; kwargs...)
    sol = DiffEqBase.solve(getfield(mp, :prob), args...; kwargs...)
    return SciMLBase.wrap_sol(sol, getfield(mp, :metadata))
end
