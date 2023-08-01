abstract type AbstractDiscretizer end

struct QuantileSlicingDiscretizer <: AbstractDiscretizer
    nbins::Int64
end

function learn_discretizations(
    disc::QuantileSlicingDiscretizer,
    Ps_train,
    Xs_train::AbstractVector{<:Real},
    Xs_test::AbstractVector{<:Real},
    α,
    multiple_testing_method,
)

    @unpack nbins = disc
    qs = [-Inf; quantile([Xs_train; Xs_test], (1:(nbins-1)) ./ nbins); +Inf]
    Xs_train_d = cut(Xs_train, qs, allowempty = true)
    Xs_test_d = cut(Xs_test, qs, allowempty = true)
    (Xs_train_d, Xs_test_d, qs)
end

struct DiscretizationWeightLearner{CW<:CategoricalWeightLearner,D<:AbstractDiscretizer} <:
       AbstractWeightLearner
    wlearner::CW
    discretizer::D
end


function learn_weights(
    dw::DiscretizationWeightLearner,
    Ps_train,
    Xs_train,
    Xs_test,
    α,
    multiple_testing_method,
)
    # TODO: check levels are the same


    Xs_train_d, Xs_test_d, discr_scheme = learn_discretizations(
        dw.discretizer,
        Ps_train,
        Xs_train,
        Xs_test,
        α,
        multiple_testing_method,
    )

    ws, ws_fit = learn_weights(
        dw.wlearner,
        Ps_train,
        Xs_train_d,
        Xs_test_d,
        α,
        multiple_testing_method,
    )
    discr_ws_fit = (discr_fit = discr_scheme, ws_fit = ws_fit)
    (ws, discr_ws_fit)
end
