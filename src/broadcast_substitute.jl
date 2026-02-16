broadcast_substitute(expr, pairs, verbose = MOLVerbosity(SciMLLogging.None())) = _sub(expr, pairs, verbose)

function _sub(expr, pairs, verbose)
    ipair = findfirst(p -> isequal(p.first, expr), pairs)
    if ipair !== nothing
        @SciMLMessage(verbose, :stencil) do
            "Replacing $expr with $(pairs[ipair])"
        end
        return pairs[ipair].second
    elseif istree(expr)
        op = operation(expr)
        args = _sub.(arguments(expr), (pairs,), (verbose,))
        if any(arg -> symtype(arg) <: AbstractArray, args) && any(!(op isa T) for T in [ArrayOp, ArrayMaker, typeof(getindex)])
            try
                return broadcast(op, args...)
                #return unwrap(op(map(wrap, args)...))
            catch e
                throw(ArgumentError("Cannot broadcast operation $op over arguments $args"))
            end
        else
            return op(args...)
        end
    else
        return expr
    end
end

function broadcast_substitute(pairs::Array{<:Pair}, verbose = MOLVerbosity(SciMLLogging.None()))
    map(pairs) do pair
        op = last(pair)
        @assert op isa ArrayOp
        args = arguments(op)
        is = args[1]
        expr = args[2]
        ranges = is .=> fist(pair)
        broadcast_substitute(expr, ranges, verbose)
    end
end