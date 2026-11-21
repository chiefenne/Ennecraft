- **Transient conjugate heat transfer (stage 7d6).** Solid regions now
  work with a `[time]` table. The fluid and the solids share one energy
  matrix with the time terms ρ c_p V/Δt of each cell's own material, so
  the interface needs no coupling iteration and has no time lag; the
  temperature's time levels follow those of the flow through adaptive,
  rejected and rewound steps. The heat balance gains the heat stored per
  region, and every transient run with the energy equation writes it per
  step to `heat.csv` (and to the run record), with the heat flow through
  each interface; the report gives the largest relative imbalance of a
  step. Probes give the temperature (`<probe>_T`), in a conjugate case on
  the whole mesh, solids included. Check V7.6: two bodies brought into
  contact keep the exact contact temperature (to 10⁻¹² K at 500 s) and
  follow the erf profiles at second order in space (2.02, 2.00) and time
  (2.05, 2.02) (`cht_contact_transient.toml`); check V7.37: the heated
  wall switched on closes its balance to 2·10⁻¹¹ in every step and ends
  at the steady solution (`cht_heated_wall_transient.toml`). Gallery and
  recipe 159.
- **`erf` and `erfc` in the expression language**, written in-house like
  the other functions (bitwise the same on every backend, within a few
  ulp of the C library), for exact solutions of transient conduction.
- **Sweeps.** A transient sweep over levels now checks the scalars with an
  exact solution (`scalar_max_error`, `scalar_order`) as a steady one
  does; a time sweep of a conjugate case takes the temperature on the
  whole mesh, and a fluid at rest no longer gives a NaN velocity order.
