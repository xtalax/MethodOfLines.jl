########################################################################################
# Stencil interface
########################################################################################

function bc_deriv(D::DerivativeOperator, ::LowerBoundaryTrait, udisc, j, is, interior, ranges)
    weights = D.low_boundary_coefs[1]
    taps = 1:D.boundary_stencil_length
    return BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
end

function bc_deriv(D::DerivativeOperator, ::UpperBoundaryTrait, udisc, j, is, interior, ranges)
    lenx = last(ranges[j])
    weights = D.high_boundary_coefs[1]
    taps = (lenx-D.boundary_stencil_length+1):lenx
    return BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
end

function bc_var(::LowerBoundaryTrait, udisc, j, is, interior, ranges)
    weights = [1]
    taps = [1]
    return BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
end

function bc_var(::UpperBoundaryTrait, udisc, j, is, interior, ranges)
    lenx = last(ranges[j])
    weights = [1]
    taps = [lenx]
    return BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
end


_is_numeric_arg(x) = let xu = safe_unwrap(x); (xu isa Number) || SymbolicUtils.isconst(xu) end

function _has_integral(eq)
    lhs = eq isa Equation ? eq.lhs : eq
    rhs = eq isa Equation ? eq.rhs : eq
    _has_integral_term(lhs) || _has_integral_term(rhs)
end

function boundary_value_rules(interior, s::DiscreteSpace{N,M,G}, boundary, derivweights) where {N,M,G<:EdgeAlignedGrid}
    u_, x_ = getvars(boundary)
    x = x_
    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc

    u = depvar(u_, s)
    args = ivs(u, s)
    j = findfirst(isequal(x_), args)

    boundary_vs = filter(v -> any(_is_numeric_arg, arguments(v)), boundary.depvars)
    non_boundary_vs = filter(v -> any(x -> !_is_numeric_arg(x), arguments(v)), boundary.depvars)

    depvarderivbcmaps = [(Differential(x_)^d)(v_) => bc_deriv(derivweights.halfoffsetmap[1][Differential(x_)^d], trait(boundary), s.discvars[u], j, get_is(v_, s), get_interior(v_, s, interior), get_ranges(v_, s)) for v_ in boundary_vs, d in derivweights.orders[x_]]

    depvarbcmaps = [v_ => bc_deriv(derivweights.interpmap[x_], trait(boundary), s.discvars[u], j, get_is(v_, s), get_interior(v_, s, interior), get_ranges(v_, s)) for v_ in boundary_vs]

    # Only make a map if the integral will actually come out to the same number of dimensions as the boundary value
    integralvs = unwrap.(filter(v -> !any(_is_numeric_arg, arguments(v)), boundary.depvars))

    integralbcmaps = _has_integral(boundary.eq) ? generate_whole_domain_integration_rules(interior, s, integralvs, indexmap, nothing, x_) : []

    if boundary isa HigherOrderInterfaceBoundary
        u__ = boundary.u2
        x__ = boundary.x2

        otherderivmaps = vec([(Differential(x__)^d)(u__) => bc_deriv(derivweights.halfoffsetmap[1][Differential(x__)^d], trait(boundary), s.discvars[depvar(u__, s)], x2i(s, u__, x__), get_is(u__, s), get_interior(u__, s, interior), get_ranges(u__, s)) for d in derivweights.orders[x_]])

        otherbcmaps = [u__ => bc_deriv(derivweights.interpmap[x__], trait(boundary), s.discvars[depvar(u__, s)], x2i(s, u__, x__), get_is(u__, s), get_interior(u__, s, interior), get_ranges(u__, s)) for u__ in boundary_vs]
        depvarderivbcmaps = vcat(depvarderivbcmaps, otherderivmaps)
        depvarbcmaps = vcat(depvarbcmaps, otherbcmaps)
    end

    varrules = varmaps(s, interior, non_boundary_vs)

    return vcat(depvarderivbcmaps, depvarbcmaps, integralbcmaps, varrules)
end

function boundary_value_rules(interior, s::DiscreteSpace{N,M,G}, boundary, derivweights) where {N,M,G<:CenterAlignedGrid}
    u_, x_ = getvars(boundary)

    x = x_
    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc

    u = depvar(u_, s)
    args = ivs(u, s)

    boundary_vs = filter(v -> any(_is_numeric_arg, arguments(v)), boundary.depvars)
    non_boundary_vs = filter(v -> any(x -> !_is_numeric_arg(x), arguments(v)), boundary.depvars)

    depvarderivbcmaps = vec([(Differential(x_)^d)(v_) => bc_deriv(derivweights.map[Differential(x_)^d], trait(boundary), s.discvars[depvar(v_, s)], x2i(s, v_, x_), get_is(v_, s), get_interior(v_, s, interior), get_ranges(v_, s)) for v_ in boundary_vs, d in derivweights.orders[x_]])

    depvarbcmaps = [v_ => bc_var(trait(boundary), s.discvars[depvar(v_, s)], x2i(s, v_, x_), get_is(v_, s), get_interior(v_, s, interior), get_ranges(v_, s)) for v_ in boundary_vs]

    # Only make a map if the integral will actually come out to the same number of dimensions as the boundary value
    integralvs = unwrap.(filter(v -> !any(_is_numeric_arg, arguments(v)), boundary.depvars))

    integralbcmaps = _has_integral(boundary.eq) ? generate_whole_domain_integration_rules(interior, s, integralvs, Dict(), nothing, x_) : []

    if boundary isa HigherOrderInterfaceBoundary
        u__ = boundary.u2
        x__ = boundary.x2
        otheru = depvar(u__, s)

        j = x2i(s, otheru, x__)

        otherderivmaps = vec([(Differential(x__)^d)(u__) => bc_deriv(derivweights.map[Differential(x__)^d], trait(boundary), s.discvars[depvar(u__, s)], j, get_is(u__, s), get_interior(u__, s, interior), get_ranges(u__, s)) for d in derivweights.orders[x_]])

        otherbcmaps = [u__ => bc_var(trait(boundary), s.discvars[depvar(u__, s)], j, get_is(u__, s), get_interior(u__, s, interior), get_ranges(u__, s)) for u__ in boundary_vs]
        depvarderivbcmaps = vcat(depvarderivbcmaps, otherderivmaps)
        depvarbcmaps = vcat(depvarbcmaps, otherbcmaps)
    end

    varrules = varmaps(s, interior, non_boundary_vs)

    return vcat(depvarderivbcmaps, depvarbcmaps, integralbcmaps, varrules)
end

function generate_bc_op_pair(s, b::AbstractTruncatingBoundary, interior, iboundary, derivweights)
    bc = b.eq

    u_, x_ = getvars(b)

    valrules = axiesvals(s, b, interior)
    is = get_is(u_, s)
    u = depvar(u_, s)
    j = x2i(s, u, x_)

    # Use full grid ranges for non-boundary dims so corners are covered in N-D.
    # Boundary at x=0 in 2D covers [1:1, 1:Ny] (not [1:1, 2:Ny-1]).
    # Always use offset(b, 1, ...) — each BC has one boundary point per side.
    # The iboundary from enumerate counts globally across dimensions, not per-dim.
    ranges = map(ivs(u_, s)) do x
        if isequal(x, x_)
            i = offset(b, 1, length(s, x))
            i:i
        else
            1:length(s, x)
        end
    end

    # Full grid ranges for BoundaryDerivArrayOp (dimensionally agnostic)
    bc_full_ranges = map(x -> 1:length(s, x), ivs(u, s))

    # boundary.depvars has general form (u(t,x)), but bc.lhs has boundary-specific
    # form (u(t,0)) with numeric args. Rules from boundary_value_rules map the general
    # form which won't match the specific form in bc.lhs.
    # Extract depvars from actual BC equation and create matching rules directly.
    depvar_ops = map(operation, s.vars.dvs)
    bc_expr = bc.lhs - bc.rhs
    bc_depvars = collect(get_depvars(bc_expr, depvar_ops))

    bc_deriv_rules = Pair[]
    bc_value_rules = Pair[]
    for bcdv in bc_depvars
        if any(_is_numeric_arg, arguments(bcdv))
            # Derivative rules for Neumann/Robin BCs (must come before value rules
            # so that Dx(u(t,0)) is matched whole, not partially via u(t,0)).
            # Orders 1 and 2 are always in derivweights.map.
            for d in unique(vcat(get(derivweights.orders, x_, Int[]), [1, 2]))
                deriv_key = Differential(x_)^d
                if haskey(derivweights.map, deriv_key)
                    push!(bc_deriv_rules, deriv_key(bcdv) => bc_deriv(
                        derivweights.map[deriv_key], trait(b),
                        s.discvars[depvar(bcdv, s)],
                        x2i(s, depvar(bcdv, s), x_),
                        get_is(bcdv, s),
                        bc_full_ranges,
                        get_ranges(bcdv, s)))
                end
            end
            # Value rule for the depvar itself (e.g., u(t, 0) → boundary value)
            push!(bc_value_rules, bcdv => bc_var(trait(b), s.discvars[depvar(bcdv, s)],
                  x2i(s, depvar(bcdv, s), x_), get_is(bcdv, s),
                  bc_full_ranges, get_ranges(bcdv, s)))
        end
    end

    # Derivative rules first, then value rules, then axis rules
    rules = vcat(bc_deriv_rules, bc_value_rules, valrules)
    val = broadcast_substitute(bc_expr, rules)

    # Fallback: if val is still scalar (e.g., constant BC not involving depvars),
    # wrap as N-D FillArrayOp to match the region dimensionality.
    if !(symtype(val) <: AbstractArray)
        value_ranges = map(r -> 1:length(r), ranges)
        val = FillArrayOp(val, Tuple(is), collect(value_ranges))
    end

    Tuple(ranges) => val
end

function generate_bc_op_pair(s, boundary::InterfaceBoundary, interior, iboundary, derivweights)
    u_, x_ = getvars(boundary)

    isupper(boundary) && return nothing
    u_ = boundary.u
    x_ = boundary.x
    u__ = boundary.u2
    x__ = boundary.x2
    N = ndims(u_, s)
    j = x2i(s, depvar(u_, s), x_)
    # * Assume that the interface BC is of the simple form u(t,0) ~ u(t,1)
    Ioffset = unitindex(N, j) * (length(s, x__) - 1)
    disc1 = s.discvars[depvar(u_, s)]
    disc2 = s.discvars[depvar(u__, s)]

    is = get_is(u_, s)
    idxs = map(1:length(is)) do i
        if i == j
            1
        else
            is[i]
        end
    end
    #I = CartesianIndex(idxs...)
    ranges = map(PDEBase.ivs(depvar(u_, s), s)) do x
        if isequal(x, x_)
            1:1
        else
            interior[x]
        end
    end

    expr = disc1[idxs...] - disc2[(idxs.+Ioffset.I)...]
    symindices = setdiff(1:ndims(u_, s), [j])

    Tuple(ranges) => FillArrayOp(expr, filter(x -> SymbolicUtils.issym(x), idxs), ranges[symindices])
end

function generate_bc_op_pairs(s, boundaries, derivweights, interior)
    lowerboundaries = sort(filter(b -> !isupper(b), boundaries), by=ordering)
    upperboundaries = sort(filter(b -> isupper(b), boundaries), by=ordering)

    lowerpairs = map(enumerate(lowerboundaries)) do (iboundary, boundary)
        generate_bc_op_pair(s, boundary, interior, iboundary, derivweights)
    end
    upperpairs = map(enumerate(upperboundaries)) do (iboundary, boundary)
        generate_bc_op_pair(s, boundary, interior, iboundary, derivweights)
    end
    return filter(pair -> pair isa Pair, vcat(lowerpairs, upperpairs))
end

#TODO: Work out Extrap Eqs
