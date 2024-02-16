# TODO: support Optim-specified bounds by overloading optimize for Fminbox{<:LBFGSB}

"""
    LBFGSB(; m::Int=10, lower=nothing, upper=nothing)

Construct a limited-memory BFGS optimizer with box constraints.

# Fields

$FIELDS
"""
struct LBFGSB{L,U} <: Optim.FirstOrderOptimizer
    "Length of the limited memory. Recommended values are 3 <= m <= 20."
    m::Int
    "Lower bound(s) for the variables. May be `nothing`, a number, or a vector of numbers."
    lower::L
    "Upper bound(s) for the variables. May be `nothing`, a number, or a vector of numbers."
    upper::U
end
LBFGSB(; m::Int=10, lower=nothing, upper=nothing) = LBFGSB(m, lower, upper)

# needed because Optim expects this field to exist, but we don't support it
function Base.getproperty(opt::LBFGSB, name::Symbol)
    name === :manifold && return Optim.Flat()
    return getfield(opt, name)
end

mutable struct LBFGSBState <: Optim.AbstractOptimizerState
    n::Int
    x::Vector{Cdouble}
    l::Vector{Cdouble}
    u::Vector{Cdouble}
    nbd::Vector{Cint}
    f::Cdouble
    g::Vector{Cdouble}
    factr::Cdouble
    pgtol::Cdouble
    wa::Vector{Cdouble}
    iwa::Vector{Cint}
    task::Vector{Cuchar}
    iprint::Cint
    csave::Vector{Cuchar}
    lsave::Vector{Cint}
    isave::Vector{Cint}
    dsave::Vector{Cdouble}
    f_x_previous::Cdouble
    x_previous::Vector{Cdouble}
    g_previous::Vector{Cdouble}
end

function Optim.initial_state(method::LBFGSB, options, obj, initial_x)
    m = method.m

    # arguments needed for setulb!
    n = length(initial_x)
    x = zeros(Cdouble, n)
    l = zeros(Cdouble, n)
    u = zeros(Cdouble, n)
    nbd = zeros(Cint, n)
    f = Cdouble(0)
    g = zeros(Cdouble, n)
    factr = pgtol = Cdouble(0)  # use Optim's own convergence criteria
    wa = zeros(Cdouble, (2m + 5) * n + 11 * m^2 + 8 * m)
    iwa = zeros(Cint, 3n)
    task = fill(Cuchar(' '), 60)
    iprint = Cint(-1)
    csave = fill(Cuchar(' '), 60)
    lsave = zeros(Cint, 4)
    isave = zeros(Cint, 44)
    dsave = zeros(Cdouble, 29)

    # caches needed by Optim
    f_x_previous = Cdouble(0)
    x_previous = zeros(Cdouble, n)
    g_previous = zeros(Cdouble, n)

    # create state
    state = LBFGSBState(
        n,
        x,
        l,
        u,
        nbd,
        f,
        g,
        factr,
        pgtol,
        wa,
        iwa,
        task,
        iprint,
        csave,
        lsave,
        isave,
        dsave,
        f_x_previous,
        x_previous,
        g_previous,
    )
    copyto!(x, initial_x)
    copyto!(x_previous, initial_x)

    # set up bounded optimization
    _set_bounds!(state, method)

    # initialize algorithm, gradient is replaced with NaNs
    copyto!(task, b"START")
    setulb!(state, method)
    view(task, 1:8) == b"FG_START" ||
        throw(ArgumentError("LBFGSB failed to initialize: $(String(copy(state.task)))"))

    # start the algorithm with valid value and gradient
    NLSolversBase.value_gradient!!(obj, x)
    state.f = Cdouble(NLSolversBase.value(obj))
    copyto!(state.g, NLSolversBase.gradient(obj))
    return state
end

function _set_bounds!(state::LBFGSBState, method::LBFGSB)
    if method.lower === nothing
        fill!(state.l, -Inf)
    else
        state.l .= Cdouble.(method.lower)
    end
    if method.upper === nothing
        fill!(state.u, Inf)
    else
        state.u .= Cdouble.(method.upper)
    end
    state.nbd .= _bounds_type.(state.l, state.u)
    return state
end

function _bounds_type(l, u)
    has_lb = l > -Inf
    has_ub = u < Inf
    if has_lb && has_ub
        return 2
    elseif has_lb
        return 1
    elseif has_ub
        return 3
    else
        return 0
    end
end

function Optim.update_state!(obj, state::LBFGSBState, method::LBFGSB)
    _update_previous!(state, obj)
    while true
        setulb!(state, method)
        if view(state.task, 1:2) == b"FG"
            _update_value_and_gradient!(state, obj)
        elseif view(state.task, 1:5) == b"NEW_X"
            return false
        elseif view(state.task, 1:5) == b"ERROR"
            @error "LBFGSB failed with error: $(String(copy(state.task)))"
            return true
        else
            @debug "LBFGSB terminated with message: $(String(copy(state.task)))"
            return true
        end
    end
end

function _update_value_and_gradient!(state::LBFGSBState, obj)
    NLSolversBase.value_gradient!(obj, state.x)
    state.f = Cdouble(NLSolversBase.value(obj))
    copyto!(state.g, NLSolversBase.gradient(obj))
    return state, obj
end

function _update_previous!(state::LBFGSBState, obj)
    state.f_x_previous = NLSolversBase.value(obj)
    copyto!(state.g_previous, NLSolversBase.gradient(obj))
    copyto!(state.x_previous, state.x)
    return state
end

function setulb!(state::LBFGSBState, method::LBFGSB)
    setulb!(
        state.n,
        method.m,
        state.x,
        state.l,
        state.u,
        state.nbd,
        state.f,
        state.g,
        state.factr,
        state.pgtol,
        state.wa,
        state.iwa,
        state.task,
        state.iprint,
        state.csave,
        state.lsave,
        state.isave,
        state.dsave,
    )
    return state
end

function Optim.trace!(
    tr, d, state::LBFGSBState, iteration, method::LBFGSB, options, curr_time=time()
)
    # adapted from Optim.common_trace!
    # https://github.com/JuliaNLSolvers/Optim.jl/blob/78ab1f4859c09ec73640829d26b3825c02b7ac95/src/utilities/trace.jl
    # MIT Expat License
    # Copyright (c) 2012: John Myles White, Tim Holy, and other contributors.
    # Copyright (c) 2016: Patrick Kofod Mogensen, John Myles White, Tim Holy, and other contributors.
    # Copyright (c) 2017: Patrick Kofod Mogensen, Asbjørn Nilsen Riseth, John Myles White, Tim Holy, and other contributors.
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(NLSolversBase.gradient(d))
    end
    g_norm = maximum(abs, NLSolversBase.gradient(d))

    # NOTE: not documented as an API function.
    return Optim.update!(
        tr,
        iteration,
        NLSolversBase.value(d),
        g_norm,
        dt,
        options.store_trace,
        options.show_trace,
        options.show_every,
        options.callback,
    )
end
