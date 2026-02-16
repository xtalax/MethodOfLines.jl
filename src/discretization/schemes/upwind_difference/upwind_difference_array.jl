########################################################################################
# Stencil interface
########################################################################################

function _upwind_difference(D, ranges, interior, is, s,
                            bs, jx, u, udisc, ispositive)
    args = ivs(u, s)

    j, x = jx
    lenx = length(s, x)
    haslower, hasupper = haslowerupper(bs, x)

    upperops = []
    lowerops = []
    if ispositive
        if !haslower
            lowerops = map(interior[j][1]:D.offside) do iboundary
                lower_boundary_deriv(D, udisc, iboundary, j, is, interior)
            end
        end
        interiorop = interior_deriv(D, udisc, s, -D.stencil_length+1:0, j, is, interior, bs)
    else
        if !hasupper
            upperops = map((lenx-D.boundary_point_count+1):interior[j][end]) do iboundary
                upper_boundary_deriv(D, udisc, iboundary, j, is, interior, lenx)
            end
        end
        interiorop = interior_deriv(D, udisc, s, 0:D.stencil_length-1, j, is, interior, bs)
    end
    boundaryoppairs = safe_vcat(lowerops, upperops)

    Construct_ArrayMaker(safe_vcat([Tuple(interior) => interiorop], boundaryoppairs))
end

"""
# upwind_difference
Generate a finite difference expression in `u` using the upwind difference at point `II::CartesianIndex`
in the direction of `x`
"""
function upwind_difference(d::Int, ranges, interior, is, s::DiscreteSpace, b, derivweights,
                           jx, u, udisc, ispositive)
    j, x = jx
    # return if this is an ODE
    ndims(u, s) == 0 && return Fill(Num(0), ())
    D = if !ispositive
        derivweights.windmap[1][Differential(x)^d]
    else
        derivweights.windmap[2][Differential(x)^d]
    end
    #@show D.stencil_coefs, D.stencil_length, D.boundary_stencil_length, D.boundary_point_count
    # unit index in direction of the derivative
    return _upwind_difference(D, ranges, interior, is, s, b, jx, u, udisc, ispositive)
end

function upwind_difference(expr, d::Int, interior, s::DiscreteSpace, b,
                           depvars, derivweights, (j, x), u, udisc, indexmap)
    is = get_is(u, s)
    uinterior = get_interior(u, s, interior)
    ndims(u, s) == 0 && return Num(0)

    D_pos = derivweights.windmap[2][Differential(x)^d]
    D_neg = derivweights.windmap[1][Differential(x)^d]

    lenx = length(s, x)
    haslower, hasupper = haslowerupper(b, x)
    grid_offsets = map(r -> first(r) - 1, uinterior)

    # Build symbolic coefficient using index variables instead of concrete grid values.
    # This keeps everything as a single symbolic expression inside an ArrayOp.
    coeff_sym = _symbolic_coeff(expr, s, u, depvars, interior, is, grid_offsets)

    # Build interior stencil expressions for both wind directions
    pos_offsets = -D_pos.stencil_length+1:0
    neg_offsets = 0:D_neg.stencil_length-1
    pos_expr = _stencil_expr(D_pos, pos_offsets, udisc, is, j, grid_offsets)
    neg_expr = _stencil_expr(D_neg, neg_offsets, udisc, is, j, grid_offsets)

    # Safe interior where both stencils are valid
    bpc_pos = D_pos.boundary_point_count
    bpc_neg = D_neg.boundary_point_count
    lo = haslower ? first(uinterior[j]) : max(first(uinterior[j]), max(bpc_pos, bpc_neg) + 1)
    hi = hasupper ? last(uinterior[j]) : min(last(uinterior[j]), lenx - max(bpc_pos, bpc_neg))
    safe_interior = collect(uinterior)
    safe_interior[j] = lo:hi
    safe_onebased = map(r -> 1:length(r), safe_interior)

    combined = IfElse.ifelse(coeff_sym > 0, coeff_sym * pos_expr, coeff_sym * neg_expr)
    interiorop = FillArrayOp(recursive_unwrap(combined), Tuple(is), safe_onebased)

    # Build boundary ops for near-boundary points where one stencil is invalid.
    # At these points, use the boundary stencil for the direction that goes out of bounds
    # and the interior stencil for the direction that's still valid.
    boundary_ops = Pair[]
    if !haslower && first(uinterior[j]) < lo
        for iboundary in first(uinterior[j]):(lo-1)
            pos_bnd = _boundary_stencil_expr(D_pos.low_boundary_coefs[iboundary],
                          1:D_pos.boundary_stencil_length, udisc, is, j)
            bnd_combined = IfElse.ifelse(coeff_sym > 0, coeff_sym * pos_bnd, coeff_sym * neg_expr)
            push!(boundary_ops, prepare_boundary_op(
                (FillArrayOp(recursive_unwrap(bnd_combined), Tuple(is),
                    map(1:length(is)) do k; k == j ? (1:1) : safe_onebased[k]; end),
                 iboundary), uinterior, j))
        end
    end
    if !hasupper && last(uinterior[j]) > hi
        for iboundary in (hi+1):last(uinterior[j])
            neg_bnd = _boundary_stencil_expr(D_neg.high_boundary_coefs[lenx - iboundary + 1],
                          (lenx-D_neg.boundary_stencil_length+1):lenx, udisc, is, j)
            bnd_combined = IfElse.ifelse(coeff_sym > 0, coeff_sym * pos_expr, coeff_sym * neg_bnd)
            push!(boundary_ops, prepare_boundary_op(
                (FillArrayOp(recursive_unwrap(bnd_combined), Tuple(is),
                    map(1:length(is)) do k; k == j ? (1:1) : safe_onebased[k]; end),
                 iboundary), uinterior, j))
        end
    end

    if isempty(boundary_ops)
        return interiorop
    else
        return Construct_ArrayMaker(safe_vcat([Tuple(safe_interior) => interiorop], boundary_ops))
    end
end

# Build a symbolic expression for the coefficient `expr` using index variables.
# Instead of substituting concrete grid values (which gives a Vector{Float64}),
# we substitute symbolic grid lookups so the result stays as a single expression
# that lives inside the ArrayOp.
function _symbolic_coeff(expr, s, u, depvars, interior, is, grid_offsets)
    sym_rules = Pair[]
    # Spatial vars → symbolic grid lookup using index variables
    for xv in ivs(u, s)
        k = x2i(s, u, xv)
        sym_val = SymbolicUtils.term(getindex, s.grid[xv], is[k] + grid_offsets[k]; type=Real)
        push!(sym_rules, xv => sym_val)
    end
    # Dependent variables → symbolic array element reference
    for v in depvars
        if ndims(v, s) > 0
            vdisc = s.discvars[v]
            vis = get_is(v, s)
            vinternal = get_interior(v, s, interior)
            voffsets = map(r -> first(r) - 1, vinternal)
            idx = ntuple(ndims(v, s)) do k
                vis[k] + voffsets[k]
            end
            push!(sym_rules, v => vdisc[idx...])
        end
    end
    broadcast_substitute(expr, sym_rules)
end

# Build the weighted stencil sum expression: Σ w_k * u[tap_k]
# D is the DerivativeOperator (needed for boundary_point_count with non-uniform dx)
function _stencil_expr(D, offsets, udisc, is, j, grid_offsets)
    taps = offsets .+ (is[j] + grid_offsets[j])
    Is = map(taps) do tap
        ntuple(ndims(udisc)) do i
            i == j ? tap : is[i] + grid_offsets[i]
        end
    end
    vals = map(I -> udisc[I...], Is)
    weights = D.stencil_coefs
    if weights isa AbstractVector && eltype(weights) <: AbstractVector
        # Non-uniform dx: weights is Vector{SVector{L,T}}, one set per interior position.
        # Use symbolic indexing to look up the correct weights at each grid point.
        stencil_idx = is[j] + grid_offsets[j] - D.boundary_point_count
        stencil_len = length(first(weights))
        weight_columns = [Float64[weights[p][k] for p in eachindex(weights)] for k in 1:stencil_len]
        sym_weights = [SymbolicUtils.term(getindex, wc, stencil_idx; type=Real) for wc in weight_columns]
        sym_dot(sym_weights, vals)
    else
        sym_dot(weights, vals)
    end
end

# Build boundary stencil sum at a fixed boundary point (taps are concrete positions)
function _boundary_stencil_expr(weights, taps, udisc, is, j)
    Is = map(taps) do tap
        ntuple(ndims(udisc)) do i
            i == j ? tap : is[i]
        end
    end
    sym_dot(weights, map(I -> udisc[I...], Is))
end

@inline function generate_winding_rules(interior, s::DiscreteSpace, depvars,
                                        derivweights::DifferentialDiscretizer, bcmap,
                                        indexmap, terms, skip = [])
    # for all independent variables and dependant variables
    rules = safe_vcat(#Catch multiplication
        reduce(safe_vcat,
               [reduce(safe_vcat,
                       [[@rule *(~~a, $(Differential(x)^d)(u), ~~b) =>
                                 upwind_difference(*(~a..., ~b...), d, interior, s,
                                                   bcmap[operation(u)][x], depvars,
                                                   derivweights, (x2i(s, u, x), x), u,
                                                   s.discvars[u], indexmap)
                          for d in (let orders = derivweights.orders[x]
                                       setdiff(orders[isodd.(orders)], skip)
                                   end)]
                         for x in ivs(u, s)], init = [])
                for u in depvars], init = []),

        #Catch division and multiplication, see issue #1
        reduce(safe_vcat,
               [reduce(safe_vcat,
                       [[@rule /(*(~~a, $(Differential(x)^d)(u), ~~b), ~c) =>
                                 upwind_difference(*(~a..., ~b...) / ~c, d, interior, s,
                                                   bcmap[operation(u)][x], depvars,
                                                   derivweights, (x2i(s, u, x), x), u,
                                                   s.discvars[u], indexmap)
                          for d in (let orders = derivweights.orders[x]
                                        setdiff(orders[isodd.(orders)], skip)

                                   end)]
                         for x in ivs(u, s)], init = [])
                for u in depvars], init = [])
    )

    wind_rules = []

    # wind_exprs = []
    for t in terms
        for r in rules
            if r(t) !== nothing
                push!(wind_rules, t => r(t))
            end
        end
    end

    return safe_vcat(wind_rules, vec(mapreduce(safe_vcat, depvars) do u
        mapreduce(safe_vcat, ivs(u, s), init = []) do x
            j = x2i(s, u, x)
            is = get_is(u, s)
            uinterior = get_interior(u, s, interior)
            uranges = get_ranges(u, s)
            let orders = derivweights.orders[x]
                oddorders = setdiff(orders[isodd.(orders)], skip)

                # for all odd orders
                if length(oddorders) > 0
                    map(oddorders) do d
                        (Differential(x)^d)(u) =>
                          upwind_difference(d, uranges, uinterior, is, s, bcmap[operation(u)][x],
                                            derivweights, (j, x), u, s.discvars[u], true)
                    end
                else
                    []
                end
            end
        end
    end))
end
