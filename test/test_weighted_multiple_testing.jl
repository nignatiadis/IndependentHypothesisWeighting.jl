using StatsBase
using MultipleTesting
using IndependentHypothesisWeighting
using Test

pvs = rand(100)
for method in [Bonferroni(), BenjaminiHochberg()]
	unit_priority_ws = PriorityWeights(ones(100))
	unit_ws = UnitWeights{Float64}(100)
	adjust_unwt = adjust(pvs, method)
	adjust_wt = adjust(pvs, unit_priority_ws, method)
	adjust_unit_wt = adjust(pvs, unit_ws, method)

	@test adjust_unwt == adjust_wt
	@test adjust_unwt == adjust_unit_wt

	half_ws = PriorityWeights([2 .* ones(50); zeros(50)])
	adjust_half_wt = adjust(pvs, half_ws, method)
	adjust_half_unwt = adjust(pvs[1:50], method)
	@test  adjust_half_wt[1:50] == adjust_half_unwt
end
