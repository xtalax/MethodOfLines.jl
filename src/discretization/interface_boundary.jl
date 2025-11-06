struct RefCartesianIndex{IType,AType,N} <: Base.AbstractCartesianIndex{N}
    I::IType
    A::AType
    RefCartesianIndex(I::IType, A=nothing) where {IType} = new{IType,typeof(A),length(I)}(I, A)
end
Base.length(IR::SymbolicUtils.BasicSymbolic{CartesianIndex}) = length(arguments(IR))
Base.length(IR::RefCartesianIndex) = length(IR.I)
Base.getindex(A::AbstractArray, IR::RefCartesianIndex) = IR.A === nothing ? A[IR.I] : IR.A[IR.I]
Base.getindex(I::RefCartesianIndex, i::Int) = RefIndex(I.A, I.I[i])

const SCartesianIndex = Union{CartesianIndex, SymbolicUtils.BasicSymbolic{<:CartesianIndex}}

Base.:+(I::RefCartesianIndex, J::SCartesianIndex) = RefCartesianIndex(I.I + J, I.A)
Base.:-(I::RefCartesianIndex, J::SCartesianIndex) = RefCartesianIndex(I.I - J, I.A)
Base.:+(I::SCartesianIndex, J::RefCartesianIndex) = RefCartesianIndex(I + J.I, J.A)
Base.:-(I::SCartesianIndex, J::RefCartesianIndex) = RefCartesianIndex(I - J.I, J.A)


(b::InterfaceBoundary)(I, s, j, isx) = wrapinterface(I, s, b, j, isx)
(b::AbstractBoundary)(I, s, j, isx) = I

# This is a bit of a hacky monad to make single dimensional grids look like arrays of the right dimension for the interface boundary conditions with the nonlinear laplacian
struct OrderedIndexArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    array::A
    index::Int
    function OrderedIndexArray(array::A, index::Int, N::Int) where {T, A<:AbstractArray{T}}
        new{T,N,A}(array, index)
    end
end

struct IfCartesianIndex{T1,T2, N} <: Base.AbstractCartesianIndex{N}
    condition::Union{Bool, SymbolicUtils.BasicSymbolic{Bool}}
    I1::T1
    I2::T2
    IfCartesianIndex(condition::Union{Bool, SymbolicUtils.BasicSymbolic{Bool}}, I1::T1, I2::T2) where {T1,T2} = new{T1,T2,length(I1)}(condition, I1, I2)
end

Base.length(I::IfCartesianIndex{<:Any,<:Any,N}) where {N} = N

Base.:+(I::IfCartesianIndex, J::SCartesianIndex) = IfCartesianIndex(I.condition, I.I1 + J, I.I2 + J)
Base.:-(I::IfCartesianIndex, J::SCartesianIndex) = IfCartesianIndex(I.condition, I.I1 - J, I.I2 - J)
Base.:+(I::SCartesianIndex, J::IfCartesianIndex) = IfCartesianIndex(I.condition, I + J.I1, I + J.I2)
Base.:-(I::SCartesianIndex, J::IfCartesianIndex) = IfCartesianIndex(I.condition, I - J.I1, I - J.I2)

Base.getindex(I::IfCartesianIndex, j::Int) = ifelse(I.condition,  I.I1[j], I.I2[j])

Base.getindex(o::OrderedIndexArray, I::SCartesianIndex) = o[I[o.index]]
Base.getindex(o::OrderedIndexArray{T,N}, is::Vararg{Int}) where {T,N} = o[CartesianIndex(is...)]
Base.getindex(o::OrderedIndexArray, I::Vararg{<:SymbolicUtils.BasicSymbolic}) = o[CartesianIndex(I...)]

struct OffsetExtendingArray{T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N}} <: AbstractArray{T,N}
    array1::A
    array2::B
    direction::Int
    offset::Int
    function OffsetExtendingArray(array1::A, array2::B, direction::Int, offset::Int) where {T, N, A<:AbstractArray{T,N}, B<:AbstractArray{T,N}}
        f(a) = Tuple(a[filter(i -> i != direction, 1:N)]...)
        s1 = f(size(array1))
        s2 = f(size(array2))
        @assert s1 == s2 "Internal Error: Arrays must be the same size in all directions except the direction of the connection, got $(s1) and $(s2)"
        new{T,N,A,B}(array1, array2, direction, offset)
    end
end

function Base.display(o::OffsetExtendingArray)
    print("OffsetExtendingArray{")
    print("array1{")
    display(o.array1)
    print("}, array2{")
    display(o.array2)
    print("}, direction{")
    print(o.direction)
    print("}, offset{")
    print(o.offset)
    print("}}\n\n")
end

function Base.show(io::IO, o::OffsetExtendingArray)
    print(io, "OffsetExtendingArray:")
    print(io, "array1:")
    display(o.array1)
    print(io, "array2:")
    display(o.array2)
    print(io, "direction:")
    print(io, o.direction)
    print(io, "offset:")
    print(io, o.offset)
    print(io, "\n")
    print(io, "\n")
end

function Base.hash(o::OffsetExtendingArray{T,N}, h::UInt) where {T,N}
    h = hash(o.array1, h)
    h = hash(o.array2, h)
    h = hash(o.direction, h)
    h = hash(o.offset, h)
    return h
end

function Base.getindex(o::OffsetExtendingArray{T,N}, I::Vararg{<:Integer}) where {T,N}
    I = [o.direction == i ? I[i] + o.offset : I[i] for i in 1:N]
    return ifelse(I[o.direction] > size(o.array1, o.direction),
        o.array1[I],
    #else
        o.array2[I])
    #end
end



function Base.getindex(o::OffsetExtendingArray{T,N}, i::SymbolicUtils.BasicSymbolic{<:Integer}, is...) where {T,N}
    I = vcat(i, is...)
    I = I + o.offset * unitindex(N, o.direction)
    ifelse(I[o.direction] > size(o.array1, o.direction),
        o.array1[I],
    #else
        o.array2[I])
    #end
end

function Base.getindex(o::OffsetExtendingArray{T,N}, I::Vararg{<:SymbolicUtils.BasicSymbolic{<:Integer}}) where {T,N}
    I = CartesianIndex(I...)
    return getindex(o, I)
end


function Base.size(o::OffsetExtendingArray{T,N}) where {T,N}
    s1 = size(o.array1)
    s2 = size(o.array2)
    s = map(1:N) do i
        if i == o.direction
            s1[i] + s2[i]
        else
            s1[i]
        end
    end
    return Tuple(s)
end

function Base.size(o::OffsetExtendingArray{T,N}, i::Int) where {T,N}
    if i == o.direction
        size(o.array1, i) + size(o.array2, i)
    else
        size(o.array1, i)
    end
end

Base.getindex(o::OffsetExtendingArray{T,N}, i::SymbolicUtils.BasicSymbolic{<:Integer}, is::Vararg{<:SymbolicUtils.BasicSymbolic{<:Integer}}) where {T,N} = SymbolicUtils.term(o, i, is...; type = T)

function bwrap(I, bs, s, j, isx=false)
    for b in bs
        I = b(I, s, j, isx)
    end
    return I
end

function bwrap(udisc::AbstractArray, bs, s, j, isx=false)
    for b in bs
        udisc = expand(udisc, b, s, j, isx)
    end
    return udisc
end

function expand(udisc::AbstractArray, b::InterfaceBoundary, s, j, isx)
    u = b.u
    u2 = b.u2
    u2disc = s.discvars[depvar(u2, s)]
    # make an offset array connecting udisc and u2disc with udisc as the base
    return _expand(udisc, u2disc, j, b)
end

function _expand(udisc, u2disc, j, b::InterfaceBoundary{Val{false}(),Val{true}()})
    return OffsetExtendingArray(udisc, u2disc, j, 0)
end

function _expand(udisc, u2disc, j, b::InterfaceBoundary{Val{true}(),Val{false}()})
    return OffsetExtendingArray(u2disc, udisc, j, size(u2disc, j))
end

function _expand(udisc, u2disc, j, b::InterfaceBoundary{B,B}) where {B}
    throw(ArgumentError("Interface $(b.eq) joins two variables at the same end of the domain, this is not supported. Please post an issue if you need this feature."))
end

@inline function wrapinterface(I::RefCartesianIndex{N,Nothing}, s::DiscreteSpace, b::InterfaceBoundary, j, isx) where {N}
    return _wrapinterface(I.I, s, b, j, isx)
end

@inline function wrapinterface(I::RefCartesianIndex, s::DiscreteSpace, ::InterfaceBoundary, j, isx)
    return I
end

@inline function wrapinterface(I, s, b::InterfaceBoundary, j, isx)
    return _wrapinterface(I, s, b, j, isx)
end

function get_interface_vars(b, s, j)
    u = b.u
    u2 = b.u2
    discu2 = s.discvars[depvar(u2, s)]
    l1 = length(s, b.x)
    l2 = length(s, b.x2)
    N = ndims(u, s)
    I1 = unitindex(N, j)
    return I1, discu2, l1, l2
end


function __wrapinterface(I, s, b::InterfaceBoundary, isupper, l1, j, isx)
    u = b.u
    u2 = b.u2
    N = ndims(u, s)
    discu2 = if isx
        OrderedIndexArray(s.grid[b.x2], j, N)
    else
        s.discvars[depvar(u2, s)]
    end
    l2 = length(s, b.x2)
    I1 = unitindex(N, j)
    # update index
    I = if isupper
        I + (1 - l1) * I1
    else
        I + (l2 - 1) * I1
    end
    return RefCartesianIndex(I, discu2)
end

function _wrapinterface(I, s, b::InterfaceBoundary{Val{false}(),Val{true}()}, j, isx)
    IfCartesianIndex(I[j] <= 1, __wrapinterface(I, s, b, false, 0, j, isx), RefCartesianIndex(I))
end

function _wrapinterface(I, s, b::InterfaceBoundary{Val{true}(),Val{false}()}, j, isx)
    l1 = length(s, b.x)
    return IfCartesianIndex(I[j] > l1,
                         __wrapinterface(I, s, b, true, l1, j, isx),
                         RefCartesianIndex(I))
end
Base.getindex(I::SymbolicUtils.BasicSymbolic{<:SCartesianIndex}, j::Int) = arguments(I)[j]
Base.length(I::SymbolicUtils.BasicSymbolic{<:SCartesianIndex})  = 

function _wrapinterface(I, s, b::InterfaceBoundary{B,B}, j, isx) where {B}
    throw(ArgumentError("Interface $(b.eq) joins two variables at the same end of the domain, this is not supported. Please post an issue if you need this feature."))
end
