Base.@kwdef struct FusedPathGrenanderLearner <: AbstractWeightLearner
    λ_multipliers::Vector{Float64} = [1.0 / 2 ^ i for i in 0:9]
end

nweightchoices(w::FusedPathGrenanderLearner) = length(w.λ_multipliers) + 1
single_or_multi(::FusedPathGrenanderLearner) = IHW.Multi()


function single_t_FDR(train_grenander, test_ms, α)
    gren_mix = MixtureModel(collect(train_grenander), collect(test_ms)/sum(test_ms))
    if pdf(gren_mix, 0.0) <= 1/α
        t = 0.0
    else
        _Fdr(t) = iszero(t) ? 1/pdf(gren_mix, 0.0) : t/cdf(gren_mix, t)
        t = find_zero(t -> _Fdr(t) - α, (0.0, 1.0))
    end
    μ_dual = pdf(gren_mix, t)/(1 - α*pdf(gren_mix, t))
    t, μ_dual
end


function invert_subgradient_tilted(
    μ,
    gr::FittedGrenanderDistribution,
    ρ,  #ρ = ρ/m_g
    b,
    α)
    μₙ = μ / (1 + α * μ)
    ρ = ρ / (1 + α * μ)
    μₚ = μₙ - ρ * b
    shifted_fs_left = gr.fs .- ρ .* gr.locs[1:(end-1)]
    shifted_fs_right = gr.fs .- ρ .* gr.locs[2:end] #gr.scratch_1
    #shifted_fs_right = #gr.scratch_2

    if μₚ >= shifted_fs_left[1]
        loc = 0.0
    elseif μₚ < shifted_fs_right[end]
        loc = 1.0
    else
        idx = searchsortedfirst(shifted_fs_left, μₚ, lt = >)
        loc_lb = gr.locs[idx]
        loc_lb_m1 = gr.locs[idx-1]
        if shifted_fs_right[idx-1] >= μₚ
            loc = loc_lb
        else
            α = (μₚ - gr.fs[idx-1] + ρ*loc_lb)/ (ρ * (loc_lb-loc_lb_m1))
            loc = α * loc_lb_m1 + (1-α) * loc_lb
        end
    end
    return loc
end


function lagrange_balance_tilted(μ, bs, ρs_prop, grs, ms, α, ::BenjaminiHochberg; linearized = false)
    ts = invert_subgradient_tilted.(μ, grs, ρs_prop, bs, α)
    Fs = IndependentHypothesisWeighting.cdf.(grs, ts)

    if linearized
        Fdr = IndependentHypothesisWeighting._Fdr_linearized(ts, Fs, ms, α)
    else
        Fdr = IndependentHypothesisWeighting._Fdr(ts, Fs, ms)
    end
    Fdr
end

function difference_matrix(n)
    Is_right = Is_left = Js_left = 1:(n-1)
    V_left = fill(-1.0, n-1)
    Js_right = 2:n
    V_right = fill(1.0, n-1)
    sparse([Is_left; Is_right], [Js_left; Js_right], [V_left; V_right])
end

function learn_weights(
    learner::FusedPathGrenanderLearner,
    Ps_train,
    Xs_train::AbstractVector{<:CategoricalValue},
    Xs_test::AbstractVector{<:CategoricalValue},
    α,
    ::BenjaminiHochberg,
)
    # check levels are the same

    λ_multipliers = learner.λ_multipliers
    nw = nweightchoices(learner)

    train_data = group(Xs_train, Ps_train)
    test_ms = groupcount(Xs_test)
    Dictionaries.sortkeys!(train_data);Dictionaries.sortkeys!(test_ms)

    prop_ms = test_ms ./ sum(test_ms)

    train_grenander = fit.(Ref(Grenander()), train_data)

    single_t, μ_dual = single_t_FDR(train_grenander, test_ms, α)

    max_density = maximum([first(gr.fs) for gr in train_grenander])

    gren_sub = pdf.(train_grenander, single_t)
    ys = collect(prop_ms .* (gren_sub .* (1+α*μ_dual) .- μ_dual))
    D = difference_matrix(length(ys))
    λmax = maximum(abs.((D*D') \ (D*ys)))

    λs = λmax .* λ_multipliers

    bs = dictionary(keys(prop_ms) .=> single_t)
    shift_iter = bs .* 0.0
    us = bs .* 0.0

    ts_unreg = unregularized_thresholds_bh(train_grenander, test_ms, α)

    ts_array = Vector{typeof(ts_unreg)}(undef, nw)
    ts_array[end] = ts_unreg

    for (i, λ) in enumerate(λs)
        ρ = λ
        ρs_prop =  ρ./ prop_ms
        for j in 1:100 # should replace this with an actual stopping rule
            myzero = find_zero( μ ->  lagrange_balance_tilted(μ,  bs, ρs_prop, train_grenander, prop_ms, α, BenjaminiHochberg())-α,
                (0., max_density))
            ts_iter =  invert_subgradient_tilted.(myzero, train_grenander, ρs_prop, bs, α)
            shift_iter = Dictionary(keys(ts_iter), coef(fit(FusedLasso, collect(ts_iter .+ us), λ/ρ)))
            us .= us .+ ts_iter .- shift_iter
            bs .= shift_iter .- us
        end
        ts_array[i] = shift_iter
    end
    ts_array
end










import Base: +, -, *

# Code below is from https://github.com/JuliaStats/Lasso.jl
# licensed under the MIT "Expat" License, Copyright (c) 2014: Simon Kornblith.

struct NormalCoefs{T}
    lin::T
    quad::T

    NormalCoefs{T}(lin::Real) where {T} = new(lin, 0)
    NormalCoefs{T}(lin::Real, quad::Real) where {T} = new(lin, quad)
end
+(a::NormalCoefs{T}, b::NormalCoefs{T}) where {T} = NormalCoefs{T}(a.lin+b.lin, a.quad+b.quad)
-(a::NormalCoefs{T}, b::NormalCoefs{T}) where {T} = NormalCoefs{T}(a.lin-b.lin, a.quad-b.quad)
+(a::NormalCoefs{T}, b::Real) where {T} = NormalCoefs{T}(a.lin+b, a.quad)
-(a::NormalCoefs{T}, b::Real) where {T} = NormalCoefs{T}(a.lin-b, a.quad)
*(a::Real, b::NormalCoefs{T}) where {T} = NormalCoefs{T}(a*b.lin, a*b.quad)

# Implements Algorithm 2 lines 8 and 19
solveforbtilde(a::NormalCoefs{T}, lhs::Real) where {T} = (lhs - a.lin)/(2 * a.quad)

# These are marginally faster than computing btilde explicitly because
# they avoid division
btilde_lt(a::NormalCoefs{T}, lhs::Real, x::Real) where {T} = lhs - a.lin > 2 * a.quad * x
btilde_gt(a::NormalCoefs{T}, lhs::Real, x::Real) where {T} = lhs - a.lin < 2 * a.quad * x

struct Knot{T,S}
    pos::T
    coefs::S
    sign::Int8
end

struct FusedLasso{T,S} <: RegressionModel
    β::Vector{T}              # Coefficients
    knots::Vector{Knot{T,S}}  # Active knots
    bp::Matrix{T}             # Backpointers
end

function StatsBase.fit(::Type{FusedLasso}, y::AbstractVector{T}, λ::Real; dofit::Bool=true) where T
    S = NormalCoefs{T}
    flsa = FusedLasso{T,S}(Array{T}(undef, length(y)), Array{Knot{T,S}}(undef, 2), Array{T}(undef, 2, length(y)-1))
    dofit && fit!(flsa, y, λ)
    flsa
end

function StatsBase.fit!(flsa::FusedLasso{T,S}, y::AbstractVector{T}, λ::Real) where {T,S}
    β = flsa.β
    knots = flsa.knots
    bp = flsa.bp

    length(y) == length(β) || throw(ArgumentError("input size $(length(y)) does not match model size $(length(β))"))

    resize!(knots, 2)
    knots[1] = Knot{T,S}(-Inf, S(0), 1)
    knots[2] = Knot{T,S}(Inf, S(0), -1)

    # Algorithm 1 lines 2-5
    @inbounds for k = 1:length(y)-1
        t1 = 0
        t2 = 0
        aminus = NormalCoefs{T}(y[k], -0.5)                # Algorithm 2 line 4
        for outer t1 = 1:length(knots)-1                         # Algorithm 2 line 5
            knot = knots[t1]
            aminus += knot.sign*knot.coefs                 # Algorithm 2 line 6
            btilde_lt(aminus, λ, knots[t1+1].pos) && break # Algorithm 2 line 7-8
        end
        bminus = solveforbtilde(aminus, λ)

        aplus = NormalCoefs{T}(y[k], -0.5)                 # Algorithm 2 line 15
        t2 = length(knots)
        while t2 >= 2                                      # Algorithm 2 line 16
            knot = knots[t2]
            aplus -= knot.sign*knot.coefs                  # Algorithm 2 line 17
            btilde_gt(aplus, -λ, knots[t2-1].pos) && break # Algorithm 2 line 18-19
            t2 -= 1
        end
        bplus = solveforbtilde(aplus, -λ)

        # Resize knots so that we have only knots[t1+1:t2-1] and 2
        # elements at either end. It would be better to use a different
        # data structure here.
        estlen = t2 - t1 + 3
        if estlen == 4
            resize!(knots, 4)
        else
            if t2 == length(knots)
                resize!(knots, t2+1)
            else
                deleteat!(knots, t2+2:length(knots))
            end
            if t1 == 1
                pushfirst!(knots, Knot{T,S}(-Inf, S(0), 1))
            else
                deleteat!(knots, 1:t1-2)
            end
        end
        knots[1] = Knot{T,S}(-Inf, S(λ), 1)                # Algorithm 2 line 28
        knots[2] = Knot{T,S}(bminus, aminus-λ, 1)          # Algorithm 2 line 29
        knots[end-1] = Knot{T,S}(bplus, aplus+λ, -1)       # Algorithm 2 line 20
        knots[end] = Knot{T,S}(Inf, S(-λ), -1)             # Algorithm 2 line 31
        bp[1, k] = bminus
        bp[2, k] = bplus
    end

    # Algorithm 1 line 6
    aminus = NormalCoefs{T}(y[end], -0.5)
    for t1 = 1:length(knots)
        knot = knots[t1]
        aminus += knot.sign*knot.coefs
        btilde_lt(aminus, 0, knots[t1+1].pos) && break
    end
    β[end] = solveforbtilde(aminus, 0)

    # Backtrace
    for k = length(y)-1:-1:1                        # Algorithm 1 line 6
        β[k] = min(bp[2, k], max(β[k+1], bp[1, k])) # Algorithm 1 line 7
    end
    flsa
end

StatsBase.coef(flsa::FusedLasso) = flsa.β
