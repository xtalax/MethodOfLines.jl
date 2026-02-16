
# Scalarized path — generic MOLMetadata
function SciMLBase.PDETimeSeriesSolution(sol::SciMLBase.AbstractODESolution{T}, metadata::MOLMetadata) where {T}
    try
        odesys = sol.prob.f.sys
        pdesys = metadata.pdesys
        discretespace = metadata.discretespace

        ivs = [discretespace.time, discretespace.ivs...]
        ivgrid = ((isequal(discretespace.time, x) ? sol.t : discretespace.grid[x] for x in ivs)...,)

        solved_unknowns = if metadata.use_ODAE
            deriv_unknowns = metadata.metadata[]
            unknowns(odesys)[deriv_unknowns]
        else
            unknowns(odesys)
        end
        # Reshape the solution to flat arrays, faster to do this eagerly.
        umap = Dict(map(discretespace.dvs) do u
            let discu = discretespace.discvars[u]
                solu = map(CartesianIndices(discu)) do I
                    i = sym_to_index(discu[I], solved_unknowns)
                    # Handle Observed
                    if i !== nothing
                        sol[i, :]
                    else
                        SciMLBase.observed(sol, safe_unwrap(discu[I]), :)
                    end
                end
                # Correct placement of time axis
                if isequal(arguments(u)[1], discretespace.time)
                    out = zeros(T, length(sol.t), size(discu)...)
                    for I in CartesianIndices(discu)
                        out[:, I] .= solu[I]
                    end
                elseif isequal(arguments(u)[end], discretespace.time)
                    out = zeros(T, size(discu)..., length(sol.t))
                    for I in CartesianIndices(discu)
                        out[I, :] .= solu[I]
                    end
                else
                    @assert false "The time variable must be the first or last argument of the dependent variable $u."
                end

                Num(u) => out
            end
        end)
        # Build Interpolations
        interp = build_interpolation(umap, ivs, ivgrid, sol, pdesys)

        return SciMLBase.PDETimeSeriesSolution{T,length(discretespace.dvs),typeof(umap),typeof(metadata),
            typeof(sol),typeof(sol.errors),typeof(sol.t),typeof(ivgrid),
            typeof(ivs),typeof(pdesys.dvs),typeof(sol.prob),typeof(sol.alg),
            typeof(interp),typeof(sol.stats)}(umap, sol, sol.errors, sol.t, ivgrid, ivs,
            pdesys.dvs, metadata, sol.prob, sol.alg,
            interp, sol.dense, sol.tslocation,
            sol.retcode, sol.stats)
    catch e
        rethrow(e)
        return sol, e
    end
end

# ArrayDiscretization path — uses pre-computed grid indices for symbolic array compatibility
function SciMLBase.PDETimeSeriesSolution(sol::SciMLBase.AbstractODESolution{T},
        metadata::MOLMetadata{hasTime, Ds, Disc, PDE, M, ArrayDiscretization}) where {T, hasTime, Ds, Disc, PDE, M}
    try
        odesys = sol.prob.f.sys
        pdesys = metadata.pdesys
        ds = metadata.discretespace

        ivs = [ds.time, ds.ivs...]
        ivgrid = ((isequal(ds.time, x) ? sol.t : ds.grid[x] for x in ivs)...,)

        solved_unknowns = unknowns(odesys)

        umap = Dict(map(ds.dvs) do u
            let discu = ds.discvars[u]
                # Use pre-computed grid indices (robust for symbolic array variables)
                grid_indices = ds.Igrid[u]
                grid_shape = size(grid_indices)

                solu = map(grid_indices) do I
                    # Splat tuple for symbolic array indexing compatibility
                    sym_expr = discu[Tuple(I)...]
                    i = sym_to_index(sym_expr, solved_unknowns)
                    if i !== nothing
                        sol[i, :]
                    else
                        SciMLBase.observed(sol, safe_unwrap(sym_expr), :)
                    end
                end

                # Correct placement of time axis
                if isequal(arguments(u)[1], ds.time)
                    out = zeros(T, length(sol.t), grid_shape...)
                    for I in grid_indices
                        out[:, I] .= solu[I]
                    end
                elseif isequal(arguments(u)[end], ds.time)
                    out = zeros(T, grid_shape..., length(sol.t))
                    for I in grid_indices
                        out[I, :] .= solu[I]
                    end
                else
                    error("Time variable must be the first or last argument of dependent variable $u")
                end

                Num(u) => out
            end
        end)

        interp = build_interpolation(umap, ivs, ivgrid, sol, pdesys)

        return SciMLBase.PDETimeSeriesSolution{T,length(ds.dvs),typeof(umap),typeof(metadata),
            typeof(sol),typeof(sol.errors),typeof(sol.t),typeof(ivgrid),
            typeof(ivs),typeof(pdesys.dvs),typeof(sol.prob),typeof(sol.alg),
            typeof(interp),typeof(sol.stats)}(umap, sol, sol.errors, sol.t, ivgrid, ivs,
            pdesys.dvs, metadata, sol.prob, sol.alg,
            interp, sol.dense, sol.tslocation,
            sol.retcode, sol.stats)
    catch e
        rethrow(e)
    end
end
