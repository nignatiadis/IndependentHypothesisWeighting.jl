using IndependentHypothesisWeighting
using Documenter

DocMeta.setdocmeta!(IndependentHypothesisWeighting, :DocTestSetup, :(using IndependentHypothesisWeighting); recursive=true)

makedocs(;
    modules=[IndependentHypothesisWeighting],
    authors="Nikos Ignatiadis <nikos.ignatiadis01@gmail.com> and contributors",
    repo="https://github.com/nignatiadis/IndependentHypothesisWeighting.jl/blob/{commit}{path}#{line}",
    sitename="IndependentHypothesisWeighting.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nignatiadis.github.io/IndependentHypothesisWeighting.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nignatiadis/IndependentHypothesisWeighting.jl",
    devbranch="main",
)
