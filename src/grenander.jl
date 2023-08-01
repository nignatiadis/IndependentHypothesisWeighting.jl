struct Grenander end

struct FittedGrenanderDistribution{T,VT<:AbstractVector{T}} <:
       ContinuousUnivariateDistribution
    fs::VT
    Fs::VT
    locs::VT
    scratch_1::VT
    scratch_2::VT
end

Distributions.@distr_support FittedGrenanderDistribution 0.0 d.locs[end]

function _grenander(pv::AbstractVector{T}) where T <: AbstractFloat
    pv_sorted = sort(pv)
    # ecdf that handles duplicated values
    pv_sorted_unique, counts = rle(pv_sorted)
    ecdf_value = cumsum(counts)
    ecdf_value = ecdf_value ./ ecdf_value[end]



    Δx = [pv_sorted_unique[1]; diff(pv_sorted_unique)]
    Δy = [ecdf_value[1]; diff(ecdf_value)]

    f = Δy ./ Δx
    f = -MultipleTesting.isotonic_regression(-f, Δx)
    F  = cumsum(f .* Δx)

    return pv_sorted_unique, f, F
end

function fit(::Grenander, ps)
    low_level_grenander_fit = _grenander(ps)
    density_rle = rle(low_level_grenander_fit[2])
    loc_idx = cumsum(density_rle[2])
    locs = [0.0; low_level_grenander_fit[1][loc_idx]]
    fs = density_rle[1]
    Fs = [0.0; cumsum(fs .* diff(locs))]
    FittedGrenanderDistribution(fs, Fs, locs, zero(fs), zero(fs))
end


function Distributions.pdf(grenander::FittedGrenanderDistribution, t::Real)
    if t > grenander.locs[end]
        return 0.0
    elseif t <= grenander.locs[1] #left-continuity
        idx = 1
    else
        idx = searchsortedfirst(grenander.locs, t) - 1
    end
    grenander.fs[idx]
end

function Distributions.cdf(grenander::FittedGrenanderDistribution, t::Real)
    if t > grenander.locs[end]
        F = 1.0
    elseif t <= grenander.locs[1] #locs[1] is 0.0, TODO: allow Dirac peak at 0?
        F = 0.0
    else
        idx = searchsortedfirst(grenander.locs, t) - 1
        loc_L = grenander.locs[idx]
        F_L = grenander.Fs[idx]
        loc_R = grenander.locs[idx+1]
        F_R = grenander.Fs[idx+1]
        λ = (t - loc_L) / (loc_R - loc_L)
        F = (1 - λ) * F_L + λ * F_R
    end
    F
end

# Similar to pdf. However, returns subgradient as interval.
function subgradient(grenander::FittedGrenanderDistribution, t::Real)
    if t > grenander.locs[end]
        lower = upper = 0.0
    elseif t <= grenander.locs[1] #left-continuity
        idx = 1
        lower = upper = grenander.fs[idx]
    else
        idx = searchsortedfirst(grenander.locs, t) - 1
        upper = grenander.fs[idx]
        if t == grenander.locs[idx + 1]
            lower = grenander.fs[idx + 1]
        else
            lower = grenander.fs[idx]
        end
    end
    Interval(lower, upper)
end

function invert_subgradient(gr::FittedGrenanderDistribution, λ)
    if λ > gr.fs[1]
        loc_lb = 0.0#return Interval(0.0, 0.0)
        loc_ub = 0.0
    elseif λ < gr.fs[end]
        loc_lb = 1.0
        loc_ub = 1.0
    else
        idx = searchsortedfirst(gr.fs, λ, lt = >)
        loc_lb = gr.locs[idx]
        loc_lb_p1 = gr.locs[idx+1]
        loc_ub = pdf(gr, loc_lb_p1) == λ ? loc_lb_p1 : loc_lb
    end
    Interval(loc_lb, loc_ub)
end

function setup_shifted_fs!(gr::FittedGrenanderDistribution, ρ)
    gr.scratch_1 .= gr.fs .- ρ .* gr.locs[1:(end-1)]
    gr.scratch_2 .= gr.fs .- ρ .* gr.locs[2:end]
end
