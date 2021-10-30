module IndependentHypothesisWeighting

using Reexport

using CategoricalArrays
using Distributions
import Distributions:cdf, pdf
using Intervals
import MLDataPattern:kfolds,FoldsView,shuffleobs

@reexport using MultipleTesting
import MultipleTesting:adjust

using Random
using SplitApplyCombine
using Statistics
using StatsBase
import StatsBase: fit, sample, weights
using Tables
using UnPack

include("grenander.jl")
include("weighted_multiple_testing.jl")
include("ihw.jl")
include("categorical_weight_learners.jl")
include("discretization_weight_learners.jl")


export PriorityWeights,
       IHW,
       GBHLearner,
       GrenanderLearner,
	   QuantileSlicingDiscretizer,
	   DiscretizationWeightLearner
end
