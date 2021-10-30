using MultipleTesting
using StatsBase
using Distributions
using CategoricalArrays
using IndependentHypothesisWeighting


Ps = rand(BetaUniformMixtureModel(0.7, 0.3),10000)


sum(adjust(Ps, BenjaminiHochberg()) .<= 0.1)


ihw_opt = IHW(folds=2, weight_learner=IndependentHypothesisWeighting.UnitWeightLearner(), α=0.1)
ihw_fit = fit(ihw_opt, Ps, rand(10000))

sum(adjust(ihw_fit) .<= 0.1)


Xs = CategoricalVector(sample(1:2, 10000))
Ps1 = rand(BetaUniformMixtureModel(0.7, 0.2),10000)
Ps2 = rand(Uniform(),10000)
Ps = Ps1 .* (Xs.==1) .+ Ps2 .* (Xs.==2)

ihw_opt = IHW(folds=2, weight_learner=GBHLearner(Storey(0.5)), α=0.1)
ihw_fit = fit(ihw_opt, Ps, Xs)

sum(adjust(ihw_fit) .<= 0.1)

sum(adjust(Ps, BenjaminiHochberg()) .<= 0.1)

ihw_grenander = IHW(folds=2, weight_learner=GrenanderLearner(), α=0.1)
ihw_grenander_fit = fit(ihw_grenander, Ps, Xs)

unique(ihw_grenander_fit.weights)
sum(adjust(ihw_grenander_fit) .<= 0.1)


ihw_discr = IHW(folds=2, weight_learner = DiscretizationWeightLearner(GrenanderLearner(), QuantileSlicingDiscretizer(2)))
Xs_cont = rand(10000) .+ convert(Vector{Float64}, Xs)

ihw_grenander_fit_discr = fit(ihw_discr, Ps, Xs_cont)


# now compare weights above to implementation based on JuMP


using JuMP
using Clp
using LinearAlgebra
using MLDataPattern

for fold_idx in [1;2]
	train_idx, test_idx = ihw_grenander_fit.kfolds[fold_idx]

	Ps_train = Ps[train_idx]
	Xs_train = Xs[train_idx]

	Xs_test = Xs[test_idx]
	m1_test = sum(Xs_test .== 1)
	m2_test = sum(Xs_test .== 2)
	ms_test = [m1_test; m2_test]
	Ps_train_1 = Ps_train[Xs_train .== 1]
	Ps_train_2 = Ps_train[Xs_train .== 2]

	gren_fit_1 = fit(IndependentHypothesisWeighting.Grenander(), Ps_train_1)
	gren_fit_2 = fit(IndependentHypothesisWeighting.Grenander(), Ps_train_2)


	mymodel = Model(Clp.Optimizer)

	@variable(mymodel, ts[1:2] >= 0)
	@variable(mymodel, Fs[1:2] >= 0)


	offsets_1 = gren_fit_1.Fs[1:(end-1)] .- gren_fit_1.fs .* gren_fit_1.locs[1:(end-1)]
	@constraint(mymodel,  Fs[1] .<= offsets_1 .+ gren_fit_1.fs .* ts[1])


	offsets_2 = gren_fit_2.Fs[1:(end-1)] .- gren_fit_2.fs .* gren_fit_2.locs[1:(end-1)]
	@constraint(mymodel,  Fs[2] .<= offsets_2 .+ gren_fit_2.fs .* ts[2])

	@constraint(mymodel, dot(ms_test, ts .- 0.1 .* Fs) <= 0)

	@objective(mymodel, Max, dot(ms_test, Fs))

	optimize!(mymodel)

	solver_ts = JuMP.value.(ts)

	solver_ws = solver_ts * sum(ms_test) ./ sum(ms_test .* solver_ts)
	@test sort(collect(ihw_grenander_fit.weighting_fits[fold_idx])) ≈ sort(solver_ws)
end
