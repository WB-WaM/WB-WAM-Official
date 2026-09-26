# Benchmark-local Native50 runtime

These modules implement the Native50 model adapter, text cache, policy loader,
and 64D-state/40D-action codec used by the HumanoidArena benchmark. They are
frozen with the benchmark so evaluation does not depend on the repository
`bridge/` package. The WB-WAM model implementation comes directly from this
checkout's `training/` package.
