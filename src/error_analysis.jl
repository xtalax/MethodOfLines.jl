function error_analysis(sys, e, verbose = MOLVerbosity())
    eqs = sys.eqs
    unknowns = sys.unknowns
    t = sys.iv
    @SciMLMessage(verbose, :error_analysis) do
        "The system of equations is:\n$eqs"
    end
    if e isa ModelingToolkit.ExtraVariablesSystemException

        rs = [Differential(t)(state) => state for state in unknowns]
        extraunknowns = [state for state in unknowns]
        extraeqs = [eq for eq in eqs]
        numderivs = 0
        for r in rs
            for eq in extraeqs
                if subsmatch(eq.lhs, r) | subsmatch(eq.rhs, r)
                    extraunknowns = vec(setdiff(extraunknowns, [r.second]))
                    extraeqs = vec(setdiff(extraeqs, [eq]))
                    numderivs += 1
                    break
                end
            end
        end
        @SciMLMessage(verbose, :error_analysis) do
            """
            There are $(length(unknowns)) variables and $(length(eqs)) equations.
            There are $numderivs time derivatives.
            The variables without time derivatives are:
            $extraunknowns
            The equations without time derivatives are:
            $extraeqs"""
        end
        rethrow(e)
    else
        rethrow(e)
    end
end
