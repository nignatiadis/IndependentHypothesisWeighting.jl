StatsBase.@weights PriorityWeights


const DirectlyWeighted = Union{Bonferroni,BenjaminiHochberg}
#TODO: assumes weights sum to length(pvals)
function adjust(pvals, ws::Union{PriorityWeights,UnitWeights}, method::DirectlyWeighted)
    weighted_pvals = copy(pvals)
    @inbounds for i = 1:length(pvals)
        weighted_pvals[i] = pvals[i] == 0.0 ? 0.0 : min(1.0, pvals[i] / ws[i])
    end
    adjust(weighted_pvals, method)
end
