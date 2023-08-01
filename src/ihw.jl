abstract type AbstractWeightLearner end
abstract type WeightLearnerFit end

struct Multi end
struct Single end

single_or_multi(::AbstractWeightLearner) = Single()

function nweightchoices(w::AbstractWeightLearner)
    nweightchoices(single_or_multi(w), w)
end

nweightchoices(::Single, w) = 1


Base.@kwdef struct IHW{M<:PValueAdjustment,WL<:AbstractWeightLearner,F,T}
    multiple_testing_method::M = BenjaminiHochberg()
    folds::F = 5
    weight_learner::WL
    α::T = 0.1
end

# interface for weight learner
function learn_weights(
    weight_learner::AbstractWeightLearner,
    Ps_train,
    Xs_train,
    Xs_test,
    α,
    multiple_testing_method,
) end

struct UnitWeightLearner <: AbstractWeightLearner end

function learn_weights(
    ::UnitWeightLearner,
    Ps_train,
    Xs_train,
    Xs_test,
    α,
    multiple_testing_method,
)
    ones(size(Xs_test, 1)), nothing
end

struct IHWResult{
    I<:IHW,
    XT,
    P,
    KF, #split into folds
    WF, #modelfits
    W
}
    pvals::P
    Xs::XT
    weights::W
    adjusted_pvals::P
    kfolds::KF
    weighting_fits::WF #Vector{}...}
    ihw_options::I
end

function fit(ihw::IHW, Ps, Xs)
    @unpack folds, multiple_testing_method, weight_learner, α = ihw

    m = length(Ps)
    kf = kfolds(shuffleobs(1:m), folds)

    if isa(single_or_multi(weight_learner), Single)
        ws = PriorityWeights(ones(m))
    else
        ws = [PriorityWeights(one(m)) for _ in Base.OneTo(nweightchoices)]
    end
    weighting_fits = []

    # Initialize MLJ machine
    for (train_idx, test_idx) in kf
        if isa(Xs, AbstractVector)
            Xs_train_view = view(Xs, train_idx)
            Xs_test_view = view(Xs, test_idx)
        else
            Xs_train_view = view(Xs, train_idx, :)
            Xs_test_view = view(Xs, test_idx, :)
        end

        #return (Ps=view(Ps, train_idx), Xs_train=Xs_train_view, Xs_test=Xs_test_view)
        tmp_wts, tmp_fit = learn_weights(
            weight_learner,
            view(Ps, train_idx),
            Xs_train_view,
            Xs_test_view,
            α,
            multiple_testing_method,
        )

        push!(weighting_fits, tmp_fit)

        if isa(single_or_multi(weight_learner), Single)
            ws[test_idx] = tmp_wts .* length(test_idx) ./ sum(tmp_wts)
        else
            for i in Base.OneTo(nweightchoices(weight_learner))
                ws[i][test_idx] = tmp_wts[i] .* length(test_idx) ./ sum(tmp_wts[i])
            end
        end
    end

    if isa(single_or_multi(weight_learner), Single)
        adj_p = MultipleTesting.adjust(Ps, ws, multiple_testing_method)
    else
        adj_p = [MultipleTesting.adjust(Ps, ws[i], multiple_testing_method) for i in Base.OneTo(length(ws))]
    end

    IHWResult(Ps, Xs, ws, adj_p, kf, weighting_fits, ihw)
end

# want to add

adjust(ihwres::IHWResult) = ihwres.adjusted_pvals
weights(ihwres::IHWResult) = ihwres.weights

rejections(ihwres::IHWResult) = rejections(ihwres, single_or_multi(ihwres.ihw_options.weight_learner))

rejections(ihwres::IHWResult, ::Single) = sum(ihwres.adjusted_pvals .<= ihwres.ihw_options.α)

function rejections(ihwres::IHWResult, ::Multi)
    n = nweightchoices(ihwres.ihw_options.weight_learner)
    [sum(ihwres.adjusted_pvals[i] .<= ihwres.ihw_options.α) for i in Base.OneTo(n)]
end
