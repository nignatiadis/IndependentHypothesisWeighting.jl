abstract type CategoricalWeightLearner <: AbstractWeightLearner end
abstract type CategoricalWeightLearnerFit <: WeightLearnerFit end

struct GBHLearner <: CategoricalWeightLearner
    pi0estimator::Pi0Estimator
end

GBHLearner() = GBHLearner(Storey(0.5))


function learn_weights(
    gbh::GBHLearner,
    Ps_train,
    Xs_train::AbstractVector{<:CategoricalValue},
    Xs_test::AbstractVector{<:CategoricalValue},
    α,
    multiple_testing_method,
)
    #TODO: check levels are the same

    function gbh_wt(Ps)
        pi0 = MultipleTesting.estimate(Ps, gbh.pi0estimator)
        (1 - pi0) / pi0
    end

    groupwise_wt = map(gbh_wt, group(Xs_train, Ps_train))
    ws = ones(size(Xs_test, 1))
    @inbounds for i = 1:size(Xs_test, 1)
        ws[i] = get(groupwise_wt, Xs_test[i], 0.0)
    end
    ws, groupwise_wt
end

struct GrenanderLearner <: CategoricalWeightLearner end

function _Fdr(ts, Fs, ms)
    num = sum(ts .* ms)
    denom = sum(Fs .* ms)
    iszero(num) ? 0.0 : num / denom
end

function _Fdr_linearized(ts, Fs, ms, α)
    num = sum(ts .* ms)
    denom = sum(Fs .* ms)
    num - α * denom
end

function lagrange_balance(λ, grs, ms, α, ::BenjaminiHochberg; linearized = false)
    ts = invert_subgradient.(grs, λ)
    ts_L = first.(ts)
    ts_R = last.(ts)
    Fs_L = cdf.(grs, ts_L)
    Fs_R = cdf.(grs, ts_R)

    if linearized
        Fdr_L = _Fdr_linearized(ts_L, Fs_L, ms, α)
        Fdr_R = _Fdr_linearized(ts_R, Fs_R, ms, α)
    else
        Fdr_L = _Fdr(ts_L, Fs_L, ms)
        Fdr_R = _Fdr(ts_R, Fs_R, ms)
    end
    # wt_sum = Interval( ((sum( ms .* first.(ts)),sum( ms .* last.(ts))) .* m ./ 0.1 ./m)...) #want this to be 1
    Interval(Fdr_L, Fdr_R)
end

function learn_weights(
    ::GrenanderLearner,
    Ps_train,
    Xs_train::AbstractVector{<:CategoricalValue},
    Xs_test::AbstractVector{<:CategoricalValue},
    α,
    ::BenjaminiHochberg,
)
    # check levels are the same


    train_data = group(Xs_train, Ps_train)
    test_ms = groupcount(Xs_test)
    train_data_view = view(train_data, keys(test_ms))
    train_grenander = fit.(Ref(Grenander()), train_data_view)

    ts_mixed = unregularized_thresholds_bh(train_grenander, test_ms, α)

    ts_mixed = sum(test_ms) .* ts_mixed ./ sum(ts_mixed .* test_ms)

    ws = ones(size(Xs_test, 1))
    @inbounds for i in eachindex(axes(Xs_test, 1))
        ws[i] = get(ts_mixed, Xs_test[i], 0.0)
    end
    ws, ts_mixed
end

function unregularized_thresholds_bh(train_grenander, test_ms, α)
    all_λs = sort(vcat([gr.fs for gr in train_grenander]...), rev = true)

    λ_idx = findfirst(
        λ -> α ∈ lagrange_balance(λ, train_grenander, test_ms, α, BenjaminiHochberg()),
        all_λs,
    )
    λ_opt = all_λs[λ_idx]

    ts = invert_subgradient.(train_grenander, λ_opt)
    ts_L = first.(ts)
    ts_R = last.(ts)

    lin_Fdr = lagrange_balance(
        λ_opt,
        train_grenander,
        test_ms,
        α,
        BenjaminiHochberg();
        linearized = true,
    )
    tmp_comb = last(lin_Fdr) ./ Intervals.span(lin_Fdr)

    tmp_comb .* ts_L .+ (1 - tmp_comb) .* ts_R
end
