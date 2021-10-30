# IndependentHypothesisWeighting

[![Build Status](https://github.com/nignatiadis/IndependentHypothesisWeighting.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nignatiadis/IndependentHypothesisWeighting.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/nignatiadis/IndependentHypothesisWeighting.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nignatiadis/IndependentHypothesisWeighting.jl)

This package provides a preliminary implementation of the Independent Hypothesis Weighting method for multiple testing with side-information, as described in:

> Ignatiadis N, Huber W (2021). “[Covariate powered cross-weighted multiple testing.](https://rss.onlinelibrary.wiley.com/doi/10.1111/rssb.12411)” Journal of the Royal Statistical Society: Series B (Statistical Methodology), 83: 720-751.

> Ignatiadis N, Klaus B, Zaugg J, Huber W (2016). “[Data-driven hypothesis weighting increases detection power in genome-scale multiple testing.](https://www.nature.com/articles/nmeth.3885)” Nature Methods. doi: 10.1038/nmeth.3885, 13: 577–580.

This package is work in progress, so that we currently recommend using the R package [IHW](https://bioconductor.org/packages/release/bioc/html/IHW.html), which is available on Bioconductor. Also please see the [MultipleTesting.jl](https://github.com/juliangehring/MultipleTesting.jl) package that provides methods for multiple testing without side-information (here we build upon the interface defined in MultipleTesting.jl).

# Example Usage


Load packages:
```julia
using Distributions
using IndependentHypothesisWeighting
using Random
using StatsBase
```

Generate synthetic data: 10,000 p-values that can be partitioned into two groups, as encoded by the side-information `Xs`.

```julia
Random.seed!(1)
Xs = CategoricalVector(sample(1:2, 10000))
Ps = rand(BetaUniformMixtureModel(0.7, 0.2), 10000) .* (Xs.==1) .+ rand(Uniform(), 10000) .* (Xs.==2) 
```
Suppose we seek to control the false discovery rate at 10\%. As a baseline, that does not use the grouping side-information,
we may run the Benjamini-Hochberg procedure:
```julia
α = 0.1
sum(adjust(Ps, BenjaminiHochberg()) .<= α ) # 580 discoveries
```
580 p-values are significant. Let us run Independent Hypothesis Weighting (IHW) with the grouping side-information:
```julia
ihw_grenander = IHW(weight_learner = GrenanderLearner(), α = α)
ihw_grenander_fit = fit(ihw_grenander, Ps, Xs)
sum(adjust(ihw_grenander_fit) .<= α) # 677 discoveries
```
IHW increased the significant discoveries to 677.
