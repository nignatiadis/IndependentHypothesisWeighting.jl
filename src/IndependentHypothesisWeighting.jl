module IndependentHypothesisWeighting

import Base.maximum

using Reexport

using CategoricalArrays
using Dictionaries
using Distributions
import Distributions: cdf, pdf
using Intervals
import MLUtils: kfolds, shuffleobs

@reexport using MultipleTesting
import MultipleTesting: adjust

using Random
using Roots
using SparseArrays
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
include("fused_grenander_learner.jl")


export PriorityWeights,
    IHW,
    GBHLearner,
    GrenanderLearner,
    QuantileSlicingDiscretizer,
    DiscretizationWeightLearner
end
