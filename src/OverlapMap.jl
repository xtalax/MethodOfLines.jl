struct OverlapMap{hasoverlap}
    map
end

function OverlapMap(map)
    if length(keys(map)) == 0
        return NoOverlap()
    else
        return OverlapMap{true}(map)
    end
end

OverlapMap() = OverlapMap{false}(Dict([]))

NoOverlap() = OverlapMap()
