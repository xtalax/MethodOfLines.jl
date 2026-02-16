function generate_ic_defaults(tconds, s::DiscreteSpace, ::MOLFiniteDifference{G, S}) where {G, S<:ScalarizedDiscretization}
    t = s.time
    if s.time !== nothing
        u0 = mapreduce(vcat, tconds) do ic
            if isupper(ic)
                throw(ArgumentError("Upper boundary condition $(ic.eq) on time variable is not supported, please use a change of variables `t => -τ` to make this an initial condition."))
            end

            args = ivs(depvar(ic.u, s), s)
            indexmap = Dict([args[i] => i for i in 1:length(args)])
            D = ic.order == 0 ? identity : (Differential(t)^ic.order)
            defaultvars = D.(s.discvars[depvar(ic.u, s)])
            broadcastable_rhs = [solve_for(ic.eq, D(ic.u))]
            out = substitute.(broadcastable_rhs, valmaps(s, depvar(ic.u, s), ic.depvars, indexmap))
            vec(defaultvars .=> substitute.(broadcastable_rhs, valmaps(s, depvar(ic.u, s), ic.depvars, indexmap)))
        end
    else
        u0 = []
    end
    return u0
end

########################################################################################
# Stencil interface
########################################################################################

function generate_ic_defaults(tconds, s::DiscreteSpace, ::MOLFiniteDifference{G, S}) where {G, S<:ArrayDiscretization}
    t = s.time
    if t !== nothing
        u0 = map(tconds) do ic
            if isupper(ic)
                throw(ArgumentError("Upper boundary condition $(ic.eq) on time variable is not supported, please use a change of variables `t => -τ` to make this an initial condition."))
            end

            u = depvar(ic.u, s)
            udisc = s.discvars[u]  # Array symbol, e.g., u(t)[1:11]
            args = ivs(u, s)
            Dt = ic.order == 0 ? identity : (Differential(t)^ic.order)
            rhs = solve_for(ic.eq, Dt(ic.u))

            # Build a Julia function from the IC expression and evaluate on the grid
            f = build_function(rhs, args...; expression=Val{false})
            grid_indices = collect(s.Igrid[u])  # N-D CartesianIndices
            grid_shape = size(grid_indices)
            vals_flat = map(vec(grid_indices)) do II
                pt = ntuple(i -> Float64(s.grid[args[i]][II[i]]), length(args))
                Float64(f(pt...))
            end
            # Reshape to match the array dimensions (N-D matrix, not flat vector)
            vals = reshape(vals_flat, grid_shape)
            # Array-level default: D(u(t)[axes...]) => vals_matrix
            # Passed directly to ODEProblem (not through System guesses)
            Dt(udisc) => vals
        end
    else
        u0 = []
    end
    return u0
end
