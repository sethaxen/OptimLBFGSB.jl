using OptimLBFGSB
using Documenter

DocMeta.setdocmeta!(OptimLBFGSB, :DocTestSetup, :(using OptimLBFGSB); recursive=true)

makedocs(;
    modules=[OptimLBFGSB],
    authors="Seth Axen <seth@sethaxen.com> and contributors",
    sitename="OptimLBFGSB.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
