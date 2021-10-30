using IndependentHypothesisWeighting
using Test



@testset "IndependentHypothesisWeighting.jl" begin
    include("test_weighted_multiple_testing.jl")
    include("test_ihw.jl")
    include("test_grenander.jl")
end
