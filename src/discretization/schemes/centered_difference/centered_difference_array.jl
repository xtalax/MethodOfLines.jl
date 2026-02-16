########################################################################################
# Stencil interface
########################################################################################

function central_difference(D::DerivativeOperator, interior, s, bs, jx, u, udisc)
    args = ivs(u, s)
    interior = get_interior(u, s, interior)
    is = get_is(u, s)

    j, x = jx
    lenx = length(s, x)
    haslower, hasupper = haslowerupper(bs, x)

    # Generate boundary stencil ops for near-boundary interior points that fall
    # within the DerivativeOperator boundary stencil zone.
    # Skip when interface boundaries exist (haslower/hasupper) — bwrap handles those.
    lowerops = []
    upperops = []

    if !haslower && first(interior[j]) <= D.boundary_point_count
        lowerops = map(first(interior[j]):D.boundary_point_count) do iboundary
            lower_boundary_deriv(D, udisc, iboundary, j, is, interior)
        end
    end
    if !hasupper && last(interior[j]) >= lenx - D.boundary_point_count + 1
        upperops = map((lenx-D.boundary_point_count+1):last(interior[j])) do iboundary
            upper_boundary_deriv(D, udisc, iboundary, j, is, interior, lenx)
        end
    end
    boundaryoppairs = safe_vcat(lowerops, upperops)

    # Narrow interior for dimension j to the stencil-valid range where the
    # centered stencil can safely access all required taps.
    # For interface boundaries (haslower/hasupper), bwrap extends the array so
    # the standard stencil can reach across — no restriction needed on that side.
    stencil_interior = collect(interior)
    lo = haslower ? first(interior[j]) : max(first(interior[j]), D.boundary_point_count + 1)
    hi = hasupper ? last(interior[j]) : min(last(interior[j]), lenx - D.boundary_point_count)
    stencil_interior[j] = lo:hi

    interiorop = interior_deriv(D, bwrap(udisc, bs, s, j, false), s, half_range(D.stencil_length), j, is, stencil_interior, bs)
    if length(boundaryoppairs) == 0
        return interiorop
    else
        return Construct_ArrayMaker(safe_vcat([Tuple(stencil_interior) => interiorop], boundaryoppairs))
    end
end

@inline function generate_cartesian_rules(interior, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, bcmap, indexmap, terms)
    return reduce(safe_vcat,
                  [reduce(safe_vcat,
                          [[(Differential(x)^d)(u) =>
                              central_difference(derivweights.map[Differential(x)^d],
                                                 interior, s, bcmap[operation(u)][x],
                                                 (x2i(s, u, x), x), u, s.discvars[u])
                             for d in (let orders = derivweights.orders[x]
                                           orders[iseven.(orders)]
                                       end)]
                           for x in ivs(u, s)], init = [])
                    for u in depvars], init = [])
end
