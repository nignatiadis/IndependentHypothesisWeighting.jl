abstract type WeightLearner end
abstract type WeightLearnerFit end


Base.@kwdef struct IHW{M<:PValueAdjustment,WL<:WeightLearner,F,T}
    multiple_testing_method::M = BenjaminiHochberg()
    folds::F = 5
    weight_learner::WL
    α::T = 0.1
end

# interface for weight learner
function learn_weights(
    weight_learner::WeightLearner,
    Ps_train,
    Xs_train,
    Xs_test,
    α,
    multiple_testing_method,
) end

struct UnitWeightLearner <: WeightLearner end

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
    KF<:FoldsView, #split into folds
    WF, #modelfits
    W<:AbstractWeights,
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

    ws = PriorityWeights(ones(m))
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
        tmp_wts, tmp_fit = learn_weights(
            weight_learner,
            view(Ps, train_idx),
            Xs_train_view,
            Xs_test_view,
            α,
            multiple_testing_method,
        )

        push!(weighting_fits, tmp_fit)
        ws[test_idx] = tmp_wts .* length(test_idx) ./ sum(tmp_wts)
    end

    adj_p = MultipleTesting.adjust(Ps, ws, multiple_testing_method)

    IHWResult(Ps, Xs, ws, adj_p, kf, weighting_fits, ihw)
end


adjust(ihwres::IHWResult) = ihwres.adjusted_pvals
weights(ihwres::IHWResult) = ihwres.weights
