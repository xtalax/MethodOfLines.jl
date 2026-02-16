########################################################################################
# Stencil interface
########################################################################################

function half_offset_centered_difference(D::DerivativeOperator, interior, s, bs, jx, u, udisc, len, isx = false)
    # Accept both Dict (raw interior map) and Vector{UnitRange} (already resolved)
    interior = interior isa AbstractVector ? interior : get_interior(u, s, interior)
    is = get_is(u, s)

    j, x = jx
    lenx = len == 0 ? length(s, x) : len

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
    # stencil can safely access all required taps.
    # For interface boundaries (haslower/hasupper), bwrap extends the array so
    # the standard stencil can reach across — no restriction needed on that side.
    stencil_interior = collect(interior)
    lo = haslower ? first(interior[j]) : max(first(interior[j]), D.boundary_point_count + 1)
    hi = hasupper ? last(interior[j]) : min(last(interior[j]), lenx - D.boundary_point_count)
    stencil_interior[j] = lo:hi

    if !isx
        interiorop = interior_deriv(D, udisc, s,
                                    (1-div(D.stencil_length, 2)):(div(D.stencil_length, 2)),
                                    j, is, stencil_interior, bs)
    else
        interiorop = interior_deriv(D, OrderedIndexArray(udisc, j, ndims(u, s)), s,
                                    (1-div(D.stencil_length, 2)):(div(D.stencil_length, 2)),
                                    j, is, stencil_interior, bs, isx)
    end

    if length(boundaryoppairs) == 0
        return interiorop
    else
        return Construct_ArrayMaker(safe_vcat([Tuple(stencil_interior) => interiorop], boundaryoppairs))
    end
end
