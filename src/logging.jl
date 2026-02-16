using SciMLLogging: SciMLLogging, @SciMLMessage, @verbosity_specifier,
    AbstractVerbositySpecifier, AbstractMessageLevel, AbstractVerbosityPreset,
    Silent, DebugLevel, InfoLevel, WarnLevel, ErrorLevel, CustomLevel,
    None, Minimal, Standard, Detailed, All, verbosity_to_int, verbosity_to_bool

SciMLLogging.@verbosity_specifier MOLVerbosity begin
    toggles = (
        :discretization,
        :grid,
        :fold,
        :stencil,
        :system_transform,
        :error_analysis,
        :solution,
        :deprecation,
        :derivative,
    )

    presets = (
        None = (
            discretization = Silent(),
            grid = Silent(),
            fold = Silent(),
            stencil = Silent(),
            system_transform = Silent(),
            error_analysis = Silent(),
            solution = Silent(),
            deprecation = Silent(),
            derivative = Silent(),
        ),
        Minimal = (
            discretization = Silent(),
            grid = WarnLevel(),
            fold = Silent(),
            stencil = Silent(),
            system_transform = Silent(),
            error_analysis = Silent(),
            solution = WarnLevel(),
            deprecation = WarnLevel(),
            derivative = Silent(),
        ),
        Standard = (
            discretization = InfoLevel(),
            grid = WarnLevel(),
            fold = Silent(),
            stencil = Silent(),
            system_transform = WarnLevel(),
            error_analysis = InfoLevel(),
            solution = WarnLevel(),
            deprecation = WarnLevel(),
            derivative = WarnLevel(),
        ),
        Detailed = (
            discretization = InfoLevel(),
            grid = InfoLevel(),
            fold = InfoLevel(),
            stencil = InfoLevel(),
            system_transform = InfoLevel(),
            error_analysis = InfoLevel(),
            solution = InfoLevel(),
            deprecation = InfoLevel(),
            derivative = InfoLevel(),
        ),
        All = (
            discretization = DebugLevel(),
            grid = DebugLevel(),
            fold = DebugLevel(),
            stencil = DebugLevel(),
            system_transform = DebugLevel(),
            error_analysis = DebugLevel(),
            solution = DebugLevel(),
            deprecation = DebugLevel(),
            derivative = DebugLevel(),
        ),
    )

    groups = (
        warnings = (:grid, :solution, :deprecation, :derivative),
        debug = (:fold, :stencil),
        progress = (:discretization, :system_transform, :error_analysis),
    )
end

const DEFAULT_MOL_VERBOSE = MOLVerbosity()

@inline _process_mol_verbose(v::MOLVerbosity) = v
@inline _process_mol_verbose(v::AbstractVerbosityPreset) = MOLVerbosity(v)
@inline _process_mol_verbose(v::Bool) = v ? DEFAULT_MOL_VERBOSE : MOLVerbosity(None())
