using MultipleTesting
using Distributions
using CategoricalArrays
using Test

mybum1 = BetaUniformMixtureModel(0.8, 0.5, 3.0)
mybum3 = BetaUniformMixtureModel(0.5, 0.3, 3.0)
mybum2 = BetaUniformMixtureModel(0.8, 0.9, 1.0)

m = 10_000
pvals1 = rand(mybum1, 10000)
pvals2 = rand(mybum2, 10000)
pvals3 = rand(mybum3, 10000)




gr1 = fit(IndependentHypothesisWeighting.Grenander(), pvals1)
gr2 = fit(IndependentHypothesisWeighting.Grenander(), pvals2)
gr3 = fit(IndependentHypothesisWeighting.Grenander(), pvals3)


@test cdf(gr1, 0.0) == 0.0
@test cdf(gr1, gr1.locs[2]) == gr1.Fs[2]
@test cdf(gr1, gr1.locs[2]/2) == gr1.Fs[2]/2
@test cdf(gr1, gr1.locs[2]/3) ≈ gr1.Fs[2]/3
@test pdf(gr1, gr1.locs[2])*gr1.locs[2] == gr1.Fs[2]
