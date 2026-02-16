########################################################################################
# Stencil interface
########################################################################################

function cartesian_nonlinear_laplacian(expr, interior, derivweights, s::DiscreteSpace, bs, depvars, x, u)
    interior = get_interior(u, s, interior)

    N = ndims(u, s)
    N == 0 && return Num(0)
    jx = j, x = (x2i(s, u, x), x)

    D_inner = derivweights.halfoffsetmap[1][Differential(x)]
    D_outer = derivweights.halfoffsetmap[2][Differential(x)]
    inner_interpolater = derivweights.interpmap[x]

    udisc = s.discvars[u]
    lenx = length(s, x)
    cliplen = lenx - 1

    # The inner derivative produces half-point values. The outer stencil needs
    # to access half-points beyond the PDE interior (boundary stencils at the
    # edges of the interior access neighboring half-points). Widen the inner
    # range to cover ALL half-point positions so the outer has enough data.
    inner_interior = collect(interior)
    inner_interior[j] = 1:cliplen

    innerderiv = half_offset_centered_difference(D_inner, inner_interior, s, bs, jx, u, udisc, lenx)
    #TODO: Proper multi dimensional interpolation - splines?
    interpvars = [v => half_offset_centered_difference(inner_interpolater, inner_interior, s, bs,
                                                       (x2i(s, v, x), x), v, s.discvars[v], lenx)
                  for v in depvars]
    # Use FULL grid for grid-value interpolation (boundary stencils need all grid points)
    full_gridpairs = map(xv -> xv => s.grid[xv], ivs(u, s))
    interpivs = map(xpair -> xpair.first => half_offset_centered_difference(inner_interpolater,
                                                                               inner_interior, s, bs, jx,
                                                                               u, xpair.second, lenx, true),
                       full_gridpairs)

    interpolated_expr = broadcast_substitute(expr, vcat(interpvars, interpivs))

    inner_arg = interpolated_expr .* innerderiv

    outerderiv = half_offset_centered_difference(D_outer, interior, s, bs, jx, u, inner_arg, cliplen)

    return outerderiv
end

@inline function generate_nonlinlap_rules(interior, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, bcmap, indexmap, terms)
    rules = reduce(safe_vcat, [vec([@rule *(~~c, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)), ~~d) => *(~c..., cartesian_nonlinear_laplacian(*(~a..., ~b...), interior, derivweights, s, bcmap[operation(u)][x], depvars, x, u), ~d...) for x in ivs(u, s)]) for u in depvars], init = [])

    rules = safe_vcat(rules, reduce(safe_vcat, [vec([@rule $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)) => cartesian_nonlinear_laplacian(*(~a..., ~b...), interior, derivweights, s, bcmap[operation(u)][x], depvars, x, u) for x in ivs(u, s)]) for u in depvars], init = []))

    rules = safe_vcat(rules, reduce(safe_vcat, [vec([@rule ($(Differential(x))($(Differential(x))(u) / ~a)) => cartesian_nonlinear_laplacian(1 / ~a, interior, derivweights, s, bcmap[operation(u)][x], depvars, x, u) for x in ivs(u, s)]) for u in depvars], init = []))

    rules = safe_vcat(rules, reduce(safe_vcat, [vec([@rule *(~~b, ($(Differential(x))($(Differential(x))(u) / ~a)), ~~c) => *(~b..., ~c..., cartesian_nonlinear_laplacian(1 / ~a, interior, derivweights, s, bcmap[operation(u)][x], depvars, x, u)) for x in ivs(u, s)]) for u in depvars], init = []))

    nonlinlap_rules = []
    for t in terms
        for r in rules
            if r(t) !== nothing
                push!(nonlinlap_rules, t => r(t))
            end
        end
    end
    return nonlinlap_rules
end
