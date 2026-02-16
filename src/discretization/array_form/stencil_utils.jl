function lower_boundary_deriv(D, udisc, iboundary, j, is, interior)
    weights = D.low_boundary_coefs[iboundary]
    taps = 1:D.boundary_stencil_length
    prepare_boundary_op((BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior),
            iboundary), interior, j)
end

function upper_boundary_deriv(D, udisc, iboundary, j, is, interior, lenx)
    weights = D.high_boundary_coefs[lenx - iboundary + 1]
    taps = (lenx-D.boundary_stencil_length+1):lenx
    prepare_boundary_op((BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior),
            iboundary), interior, j)
end

function integral_op_pair(dx, udisc, j, is, interior, i)
    prepare_boundary_op((IntegralArrayOp(dx, udisc, i, j, is, interior),
            i), interior, j)
end

function prepare_boundary_op(boundaryop, interior, j)
    function maketuple(i)
        out = map(1:length(interior)) do k
            k == j ? (i isa Integer ? (i:i) : i) : interior[k]
        end
        return Tuple(out)
    end
    (op, iboundary) = boundaryop
    return maketuple(iboundary) => op
end

function prepare_boundary_ops(boundaryops, interior, j)
    function maketuple(i)
        out = map(1:length(interior)) do k
            k == j ? (i isa Integer ? (i:i) : i) : interior[k]
        end
        return Tuple(out)
    end
    return map(boundaryops) do (op, iboundary)
        maketuple(iboundary) => op
    end
end

function interior_deriv(D::DerivativeOperator{T,N,Wind,DX}, udisc, s, offsets, j, is, interior, bs, isx = false) where {T,N,Wind,DX<:Number}
    weights = D.stencil_coefs
    # ArrayOp shapes are always 1-based. Offset taps to map from 1-based
    # iteration index to actual grid position.
    grid_offsets = map(r -> first(r) - 1, interior)
    taps = offsets .+ (is[j] + grid_offsets[j])
    onebased_interior = map(r -> 1:length(r), interior)
    InteriorDerivArrayOp(weights, taps, udisc, s, j, is, onebased_interior, bs, isx, grid_offsets)
end

function interior_deriv(D::DerivativeOperator{T,N,Wind,DX}, udisc, s, offsets, j, is, interior, bs, isx = false) where {T,N,Wind,DX<:AbstractVector}
    @assert !any(b -> b isa AbstractInterfaceBoundary, bs) "Interface boundary conditions are not yet supported for nonuniform dx dimensions, please post an issue to https://github.com/SciML/MethodOfLines.jl if you need this functionality."
    grid_offsets = map(r -> first(r) - 1, interior)
    # For non-uniform dx, stencil_coefs is a Vector{SVector{L,T}} (one set of weights
    # per interior position). Extract column vectors so each weight can be looked up
    # symbolically at the current grid index.
    stencil_idx = is[j] + grid_offsets[j] - D.boundary_point_count
    stencil_len = length(first(D.stencil_coefs))
    weight_columns = [Float64[D.stencil_coefs[p][k] for p in eachindex(D.stencil_coefs)] for k in 1:stencil_len]
    weights = [SymbolicUtils.term(getindex, wc, stencil_idx; type=Real) for wc in weight_columns]
    taps = offsets .+ (is[j] + grid_offsets[j])
    onebased_interior = map(r -> 1:length(r), interior)
    InteriorDerivArrayOp(weights, taps, udisc, s, j, is, onebased_interior, bs, isx, grid_offsets)
end

function BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
    Is = map(taps) do tap
        map(1:ndims(udisc)) do i
            if i == j
                tap
            else
                is[i]
            end
        end
    end
    expr = sym_dot(weights, map(I -> udisc[I...], Is))
    # Keep all N dimensions: boundary dim j gets 1:1 range.
    # The expression doesn't use is[j], so it broadcasts over that dimension.
    full_interior = map(1:length(is)) do i
        i == j ? (1:1) : interior[i]
    end
    return FillArrayOp(recursive_unwrap(expr), Tuple(is), full_interior)
end

reduce_interior(interior, j) = first(interior[j]) == 1 ? [interior[1:j-1]..., 2:last(interior[j]), interior[j+1:end]...] : interior

function trapezium_sum(interior, dx, udisc, is, j)
    N = ndims(udisc)
    i = is[j]
    im1 = is[j] - 1
    Im1 = ntuple(k -> k == j ? im1 : is[k], N)
    I = ntuple(k -> is[k], N)
    rinterior = reduce_interior(interior, j)
    if dx isa Number
        expr = (dx*(udisc[Im1...] + udisc[I...]) / 2)
    else
        dx_im1 = SymbolicUtils.term(getindex, dx, im1; type=Real)
        dx_i = SymbolicUtils.term(getindex, dx, i; type=Real)
        expr = (dx_im1*udisc[Im1...] + dx_i*udisc[I...]) / 2
    end

    return FillArrayOp(expr, is, rinterior)
end

function IntegralArrayOp(dx, udisc, k, j, is, interior, iswd = false)
    if iswd
        interior = [interior[1:j-1]..., 1:size(udisc, j), interior[j+1:end]...]
    end
    rinterior = reduce_interior(interior, j)
    # Build cumulative sum manually since sum(ArrayOp) is not supported
    # trapezium_sum produces expression for (dx*(u[i-1] + u[i])/2) at each point
    # We accumulate from first(rinterior[j]) to k
    N = ndims(udisc)
    i_sym = is[j]
    start = first(rinterior[j])
    acc = Num(0)
    for idx in start:k
        im1 = idx - 1
        Im1 = ntuple(d -> d == j ? im1 : is[d], N)
        I = ntuple(d -> d == j ? idx : is[d], N)
        if dx isa Number
            acc = acc + dx * (udisc[Im1...] + udisc[I...]) / 2
        else
            dx_im1 = SymbolicUtils.term(getindex, dx, im1; type=Real)
            dx_i = SymbolicUtils.term(getindex, dx, idx; type=Real)
            acc = acc + (dx_im1 * udisc[Im1...] + dx_i * udisc[I...]) / 2
        end
    end
    symindices = setdiff(1:N, [j])
    return FillArrayOp(acc, is[symindices], rinterior[symindices])
end

function IntegralArrayMaker(dx, udisc, k, j, is, ranges, interior, iswd = false)
    if iswd
        ranges = [ranges[1:j-1]..., 1:size(udisc, j), ranges[j+1:end]...]
    end

    return IntegralArrayOp(dx, udisc, k, j, is, interior, iswd)
end

function InteriorDerivArrayOp(weights, taps, udisc, s, j, output_idx, interior, bs, isx = false, grid_offsets = nothing)
    Is = map(taps) do tap
        ntuple(ndims(udisc)) do i
            if i == j
                tap  # Already offset by grid_offsets[j] in interior_deriv
            else
                # Offset 1-based iteration index to actual grid position
                grid_offsets === nothing ? output_idx[i] : output_idx[i] + grid_offsets[i]
            end
        end
    end
    expr = sym_dot(weights, map(I -> udisc[I...], Is))

    return FillArrayOp(expr, output_idx, interior)
end

function FillArrayOp(expr, output_idx, interior)
    # ArrayOp requires a BasicSymbolic expression; wrap concrete numbers
    if !(expr isa SymbolicUtils.BasicSymbolic)
        expr = Symbolics.unwrap(Num(expr))
    end
    ranges = Dict(output_idx .=> interior)
    return ArrayOp{SymReal}(output_idx, expr, +, nothing, ranges)
end

NullBG_ArrayMaker(ranges, ops) = ArrayMaker{SymReal}([ranges, map(p -> p.first, ops)...],
                                                  [fill(0, last.(ranges)...), map(p -> p.second, ops)...])

Construct_ArrayMaker(ops) = ArrayMaker{SymReal}(map(p -> p.first, ops), map(p -> p.second, ops))

function get_interior(u, s, interior::AbstractDict)
    map(ivs(u, s)) do x
        if haskey(interior, x)
            interior[x]
        else
            1:length(s, x)
        end
    end
end

# Already-converted interior (Vector of ranges) — return as-is
get_interior(u, s, interior::AbstractVector) = interior

function get_ranges(u, s)
    map(x -> first(axes(s.grid[x])), ivs(u, s))
end
get_is(u, s) = map(x -> s.index_syms[x], ivs(u, s))
