# This os a disgusting hack to get around the fact that ArrayMakers are not supported for codegen

isarr(x) = symtype(x) isa AbstractArray

#TODO missing primitives for broadcasting over broadcasted objects

#* Assuming no nested ArrayMakers
#* assuming no mapreduce objects

#! Unfinished

"""
Should fold any broadcasts to inner ArrayOps, and fold inner ArrayOps
"""
function fold(term, verbose = MOLVerbosity(SciMLLogging.None()))
    if !istree(term)
        return term
    end
    args = arguments(term)
 #   try
        if term isa ArrayMaker
            @SciMLMessage(verbose, :fold) do; "args = $args"; end
            tterm = nothing
            ipairs = []
            for (i, arg) in enumerate(args[4:end])
                if arg isa ArrayMaker
                    _args = arguments(arg)
                    tterm = arg
                    push!(ipairs, i => (_args[3] .=> fold.(_args[4:end], (verbose,))))
                end
            end
            @SciMLMessage(verbose, :fold) do; "pairs = $ipairs"; end
            #TODO Check that this is robust to small arraymakers being split in to larger ones
            for (i, pair) in ipairs
                # flatten Arraymakers
                ranges = map(p -> p[1], pair)
                ops = map(p -> p[2], pair)
                # insert pairs keeping track of the offset
                if !isnothing(term) && size(term) != size(tterm)
                    args[3] = vcat(args[3][1:i-1], map(r -> r .+ map(_r -> _r[1], args[3][i]), ranges), args[3][i+1:end])
                else
                    args[3] = vcat(args[3][1:i-1], ranges, args[3][i+1:end])
                end
                args[4:end] = vcat(args[4:end][1:i-1], ops, args[4:end][i+1:end])
            end

            T = args[1]
            pairs = args[3] .=> fold.(args[4:end], (verbose,))
            out = Construct_ArrayMaker(pairs)
            if any(hasarraymaker, args[4:end])
                return fold(out, verbose)
            else
                return out
            end
        end
        op = operation(term)
        if term isa ArrayOp
                return op(fold.(args, (verbose,))...)

        elseif any(x -> !(op isa x), [typeof(getindex)])
            if length(arguments(term)) > 1
                return broadcast_reduce(op, fold.(arguments(term), (verbose,))..., verbose)

            elseif length(arguments(term)) == 1
                return op(fold.(arguments(term), (verbose,))[1])
            end
        else
            return term
        end
  #=  catch e
        println("Faliure with term:")
        @show term
        rethrow(e)
    end =#
end
   

const SNumber = Union{Number, SymbolicUtils.BasicSymbolic{SymbolicUtils.SymReal}}

function broadcast_reduce(f, a::SNumber, b::ArrayMaker, verbose = MOLVerbosity(SciMLLogging.None()))
    args = arguments(b)
    T = args[1]

    pairs = args[3] .=> broadcast_reduce.((f,), (a,), fold.(args[4:end], (verbose,)), verbose)
    return Construct_ArrayMaker(pairs)
end

function broadcast_reduce(f, a::ArrayMaker, b::SNumber, verbose = MOLVerbosity(SciMLLogging.None()))
    args = arguments(a)
    T = args[1]

    pairs = args[3] .=> broadcast_reduce.((f,), fold.(args[4:end], (verbose,)), (b,), verbose)
    return Construct_ArrayMaker(pairs)
end

function broadcast_reduce(f, a::ArrayMaker, b::ArrayMaker, verbose = MOLVerbosity(SciMLLogging.None()))
    args1 = arguments(a)
    args2 = arguments(b)
    T = promote_type(args1[1], args2[1])
    @assert args1[2] == args2[2] "Dimension mismatch: sizes of the following ArrayMakers are not equal: $a and $b"
    pairs1 = args1[3] .=> fold.(args1[4:end], (verbose,))
    pairs2 = args2[3] .=> fold.(args2[4:end], (verbose,))
    pairs = broadcast_reduce(f, pairs1, pairs2, verbose)
    return Construct_ArrayMaker(pairs)
end

function broadcast_reduce(f, a::Vector{<:Pair}, b::Vector{<:Pair}, verbose = MOLVerbosity(SciMLLogging.None()))
    rangesa, opsa = a
    rangesb, opsb = b
    ranges = fold_ranges(rangesa, rangesb; verbose=verbose)
    pairs = map(ranges) do r
        r.region => broadcast_reduce(f, fold.(opsa[r.A_idxs], (verbose,)), fold.(opsb[r.B_idxs], (verbose,)), verbose)
    end
    return pairs
end

function broadcast_reduce(f, a::ArrayOp, b::ArrayOp, verbose = MOLVerbosity(SciMLLogging.None()))
    args1 = arguments(a)
    args2 = arguments(b)
    is1 = args1[1]
    is2 = args2[1]
    @SciMLMessage(verbose, :fold) do; "is1 = $is1, is2 = $is2"; end
    expr = broadcast_reduce(f, fold.(args1[2], (verbose,)), fold.(args2[2], (verbose,)), verbose)
    @SciMLMessage(verbose, :fold) do; "expr = $expr"; end
    @assert all(isequal.(is1, is2)) "reducing indices different for $a and $b, got $is1 and $is2"
    @SciMLMessage("all(isequal.(is1, is2)) = true", verbose, :fold)
    @assert args1[3] == args2[3] "reducing ops different for $a and $b"
    @SciMLMessage("args1[3] == args2[3] = true", verbose, :fold)
    ranges = _intersection.(map(i_x -> args1[6][i_x], is1), map(i_x -> args2[6][i_x], is2))

    @SciMLMessage(verbose, :fold) do; "ranges = $ranges"; end
    return FillArrayOp(expr, is, ranges)
end

function broadcast_reduce(f, a::SNumber, b::ArrayOp, verbose = MOLVerbosity(SciMLLogging.None()))
    args = arguments(b)
    is = args[1]
    expr = f(a, args[2])
    @SciMLMessage(verbose, :fold) do; "expr = $expr"; end
    ranges = map(i_x -> args[6][i_x], is)

    @SciMLMessage(verbose, :fold) do; "ranges = $ranges"; end
    return FillArrayOp(expr, is, ranges)
end

function broadcast_reduce(f, a::ArrayOp, b::Number, verbose = MOLVerbosity(SciMLLogging.None()))
    args = arguments(a)
    is = args[1]
    expr = f(args[2], b)
    ranges = map(i_x -> args[6][i_x], is)

    @SciMLMessage(verbose, :fold) do; "ranges = $ranges"; end
    return FillArrayOp(expr, is, ranges)
end

function broadcast_reduce(f, a::ArrayMaker, b::ArrayOp, verbose = MOLVerbosity(SciMLLogging.None()))
    args = arguments(a)
    pairs = args[3] .=> broadcast_reduce(f, map(i -> (fold.(args[4:end][i], verbose), fold.(b[args[3][i]], verbose)), 1:length(args[3]))..., verbose)

    @SciMLMessage(verbose, :fold) do; "pairs = $pairs"; end
    return Construct_ArrayMaker(pairs)
end

function broadcast_reduce(f, a::ArrayOp, b::ArrayMaker, verbose = MOLVerbosity(SciMLLogging.None()))
    args = arguments(b)
    pairs = args[3] .=> broadcast_reduce(f, map(i -> (fold.(a[args[3][i]], verbose), fold.(args[4:end][i], verbose)), 1:length(args[3]))..., verbose)

    @SciMLMessage(verbose, :fold) do; "pairs = $pairs"; end
    return Construct_ArrayMaker(pairs)
end

broadcast_reduce(f, a::SNumber, b::SNumber, verbose = MOLVerbosity(SciMLLogging.None())) = f(a, b)

function fold_ranges(A::Vector{<:Pair}, B::Vector{<:Pair};
                                    keep_pairs::Bool=true, sort_output::Bool=true, verbose = MOLVerbosity(SciMLLogging.None()))
    rangesa, opsa = A
    rangesb, opsb = B
    ranges = fold_ranges(rangesa, rangesb; verbose=verbose)
    ranges = ranges .=> (opsa[ranges.A_idxs], opsb[ranges.B_idxs])
    return pairs
end


# --- helpers ---
_normalize(r::UnitRange{<:Integer}) = r
_normalize(i::Integer) = i:i

# closed integer range intersection; `nothing` if empty
function _intersect(r1::UnitRange{<:Integer}, r2::UnitRange{<:Integer})
    a = max(first(r1), first(r2))
    b = min(last(r1),  last(r2))
    a <= b ? (a:b) : nothing
end


# for sorting/keys: get (lo,hi) even if it's an Int
_bounds(x::Integer) = (x, x)
_bounds(r::UnitRange{<:Integer}) = (first(r), last(r))

function fold_ranges(A::Vector{<:NamedTuple}, B::Vector{<:NamedTuple};
                                    keep_pairs::Bool=true, sort_output::Bool=true, verbose = MOLVerbosity(SciMLLogging.None()))
    return fold_ranges(map(x -> x.region, A), map(x -> x.region, B); keep_pairs=keep_pairs, sort_output=sort_output, verbose=verbose)
end


# --- main: N-D with provenance ---
"""
    fold_ranges(A, B; keep_pairs=true, sort_output=true)

Return a vector of NamedTuples:
    (region = ::Tuple, A_idxs = ::Vector{Int}, B_idxs = ::Vector{Int},
     pairs = ::Vector{Tuple{Int,Int}})

Each `region` is an N-tuple of Ints or UnitRanges. `A_idxs`/`B_idxs` list all
indices of A/B that contributed to that region. `pairs` lists all (i,j) pairs
(if `keep_pairs=true`).

`sort_output`: if true, order regions lexicographically by (lo,hi) per dimension.
"""
function fold_ranges(A::Vector{<:Tuple}, B::Vector{<:Tuple};
                                    keep_pairs::Bool=true, sort_output::Bool=true, verbose = MOLVerbosity(SciMLLogging.None()))
    # Dict: region_tuple => (Set of A idxs, Set of B idxs, Set of pairs)
    prov = Dict{Tuple, Tuple{Set{Int}, Set{Int}, Set{Tuple{Int,Int}}}}()

    for (ia, a) in enumerate(A)
        @SciMLMessage(verbose, :fold) do; "A[$ia] = $a"; end
        an = map(_normalize, a)
        for (ib, b) in enumerate(B)
            @SciMLMessage(verbose, :fold) do; "B[$ib] = $b"; end
            bn = map(_normalize, b)
            length(an) == length(bn) || throw(ArgumentError("A[$ia] and B[$ib] have different dimensionality"))
            ints = map(_intersect, an, bn)
            @SciMLMessage(verbose, :fold) do; "ints = $ints"; end
            any(x -> x === nothing, ints) && continue
            region = ints |> Tuple

            if haskey(prov, region)
                @SciMLMessage("haskey(prov, region) = true", verbose, :fold)
                Sa, Sb, Sp = prov[region]
                push!(Sa, ia); push!(Sb, ib)
                keep_pairs && push!(Sp, (ia, ib))
            else
                @SciMLMessage("haskey(prov, region) = false", verbose, :fold)
                Sa = Set([ia]); Sb = Set([ib])
                Sp = Set{Tuple{Int,Int}}()
                keep_pairs && push!(Sp, (ia, ib))
                prov[region] = (Sa, Sb, Sp)
            end
        end
    end

    out = Vector{NamedTuple}(undef, length(prov))
    i = 1
    for (region, (Sa, Sb, Sp)) in prov
        @SciMLMessage(verbose, :fold) do; "region = $region"; end
        out[i] = (
            region = region,
            A_idxs = sort!(collect(Sa)),
            B_idxs = sort!(collect(Sb)),
            pairs  = keep_pairs ? sort!(collect(Sp)) : Tuple{Int,Int}[]
        )
        i += 1
    end

    @SciMLMessage(verbose, :fold) do; "out = $out"; end
    if sort_output
        # lexicographic by (lo,hi) per dimension
        keyfn = nt -> map(_bounds, nt.region) |> collect
        sort!(out; by = keyfn)
    end

    return out
end

function _intersection(r1::AbstractRange, r2::AbstractRange)
    r = max(r1[1], r2[1]):min(r1[end], r2[end])
    if length(r) == 0
        return nothing
    else
        return r
    end
end

function _intersection(p1::Pair, p2::Pair)
    r = _union(p1[1], p2[1])
    if r === nothing
        return nothing
    else
        return r => vcat(p1[2], p2[2])
    end
end

function _union(r1::AbstractRange, r2::AbstractRange)
    r = min(r1[1], r2[1]):max(r1[end], r2[end])
    if length(r) == 0
        return nothing
    else
        return r
    end
end

function _union(p1::Pair, p2::Pair)
    r = _union(p1[1], p2[1])
    if r === nothing
        return nothing
    else
        return r => vcat(p1[2], p2[2])
    end
end
function hasarraymaker(term)
    if istree(term)
        return any(hasarraymaker, arguments(term))
    elseif term isa ArrayMaker
        return true
    else
        return false
    end
end