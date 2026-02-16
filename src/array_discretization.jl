function PDEBase.generate_metadata(s::DiscreteSpace, disc::MOLFiniteDifference{G, D},
        pdesys, boundarymap, complexmap) where {G, D <: ArrayDiscretization}
    use_ODAE = false
    return MOLMetadata(s, disc, pdesys, use_ODAE)
end

# Override discretize for ArrayDiscretization to wrap result in MOLPDEProblem,
# enabling automatic PDESolution wrapping after solve.
function SciMLBase.discretize(pdesys::PDESystem,
                              discretization::MOLFiniteDifference{G, ArrayDiscretization};
                              analytic = nothing, kwargs...) where {G}
    sys, tspan, u0 = SciMLBase.symbolic_discretize(pdesys, discretization)
    u0_pairs = u0 === nothing ? Pair[] : u0
    try
        simpsys = mtkcompile(sys)
        disc_meta = get_disc_metadata(sys)
        if tspan === nothing
            add_metadata!(disc_meta, sys)
            prob = NonlinearProblem(simpsys, ones(length(get_eqs(simpsys)));
                                   discretization.kwargs..., kwargs...)
            return MOLPDEProblem(prob, disc_meta)
        else
            add_metadata!(get_disc_metadata(simpsys), sys)
            prob = ODEProblem(simpsys, u0_pairs, tspan; build_initializeprob=false,
                             discretization.kwargs..., kwargs...)
            if analytic !== nothing
                f = ODEFunction(pdesys, discretization; analytic = analytic,
                               discretization.kwargs..., kwargs...)
                prob = ODEProblem(f, prob.u0, prob.tspan, prob.p;
                                 discretization.kwargs..., kwargs...)
            end
            return MOLPDEProblem(prob, disc_meta)
        end
    catch e
        PDEBase.error_analysis(sys, e)
    end
end

PDEBase.get_discvars(s::DiscreteSpace) = s.discvars

function PDEBase.discretize_equation!(
    disc_state::PDEBase.EquationState, pde::Equation, interiormap,
    eqvar, bcmap, depvars, s::DiscreteSpace, derivweights, indexmap,
    discretization::MOLFiniteDifference{G, D}) where {G, D <: ArrayDiscretization}
    verbose = discretization.verbose

    # Pure ODE variable (no spatial dimensions) — pass through without ArrayMaker.
    if ndims(eqvar, s) == 0
        push!(disc_state.eqs, pde)
        return
    end

    # Handle boundary values appearing in the equation by creating functions that map each point on the interior to the correct replacement rule

    # Find boundaries for this equation
    eqvarbcs = mapreduce(x -> bcmap[operation(eqvar)][x], vcat, s.ivs)
    # Extract Interior
    interior = interiormap.I[pde]

    # Generate the boundary conditions for the correct variable
    boundary_op_pairs = generate_bc_op_pairs(s, eqvarbcs, derivweights, interior)
    boundary_rules = mapreduce(b -> boundary_value_rules(interior, s, b, derivweights),
                               safe_vcat, filter_extending(flatten_vardict(bcmap)), init = [])

    # Generate the discrete form ODEs for the interior
    pdeinterior = begin
        rules = vcat(generate_finite_difference_rules(interior, s, depvars, pde,
                                                      derivweights, bcmap, indexmap),
                     boundary_rules,
                     arrayvalmaps(s, eqvar, depvars, interior))
        @SciMLMessage(verbose, :discretization) do
            "Schemes Applied: The following rules were applied for the PDE $pde with the var $eqvar:"
        end
        try
            broadcast_substitute(pde.lhs, rules, verbose)
            #broadcast_substitute(pde.lhs, rules, verbose)
        catch e
            @SciMLMessage(verbose, :discretization) do
                "A scheme has been incorrectly applied to the following equation: $pde."
            end
            #println("The following rules were constructed:")
            #display(rules)
            rethrow(e)
        end
    end
    interior = get_interior(eqvar, s, interior)
    ranges = get_ranges(eqvar, s)
    bg = fill(0, last.(ranges)...)
    eqarray = deepcopy(bg) ~ ArrayMaker{SymReal}([ranges, interior, map(p -> p.first, boundary_op_pairs)...],
                                                 [bg, pdeinterior, map(p -> p.second, boundary_op_pairs)...])

    push!(disc_state.eqs, eqarray)
end

function safe_show(term, verbose = MOLVerbosity(SciMLLogging.None()))
    if istree(term)
        args = arguments(term)
        @SciMLMessage(verbose, :stencil) do
            "operation: $(operation(term))"
        end
        for (i, arg) in enumerate(args)
            try
                @SciMLMessage(verbose, :stencil) do
                    "arg $i: $arg"
                end
            catch e
                @SciMLMessage("Failure with argument $i", verbose, :stencil)
                safe_show.(args, (verbose,))
            end
        end
    else
        @SciMLMessage(verbose, :stencil) do
            "term: $term"
        end
    end
end

# function generate_system(alleqs, bceqs, ics, discvars, defaults, ps, tspan, metadata)
#     t = metadata.discretespace.time
#     name = metadata.pdesys.name
#     bceqs = reduce(vcat, bceqs)
#     alleqs = reduce(vcat, alleqs)
#     alleqs = vcat(alleqs, unique(bceqs))
#     alldepvarsdisc = vec(reduce(vcat, vec(unique(reduce(vcat, vec.(values(discvars)))))))
#     # Finalize
#     try
#         if t === nothing
#             # At the time of writing, NonlinearProblems require that the system of equations be in this form:
#             # 0 ~ ...
#             # Thus, before creating a NonlinearSystem we normalize the equations s.t. the lhs is zero.
#             sys = NonlinearSystem(eqs, alldepvarsdisc, ps, defaults=defaults, name=name, metadata=metadata)
#             return sys, nothing
#         else
#             # * In the end we have reduced the problem to a system of equations in terms of Dt that can be solved by an ODE solver.

#             sys = ODESystem(alleqs, t, alldepvarsdisc, ps, defaults=defaults, name=name, metadata=metadata)
#             return sys, tspan
#         end
#     catch e
#         println("The system of equations is:")
#         println(alleqs)
#         println()
#         println("Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.")
#         println()
#         rethrow(e)
#     end
# end
