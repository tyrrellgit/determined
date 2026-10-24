# determined
A lightweight library for some basic filtering methods

## Documentation

This crate implements small, statically-sized Kalman and Extended Kalman Filter
implementations using nalgebra's `SMatrix` for predictable, allocation-free
linear algebra.

Generate API documentation locally with:

```bash
cargo doc --open
```

The main modules are:
- `crate::common` — core traits and type aliases (includes `Algorithm` trait).
- `crate::algorithms` — Kalman and EKF implementations.
- `crate::state` — small `State<R,C>` wrapper around nalgebra matrices.
- `crate::measurement` — measurement/observation helpers.
- `crate::models` — traits for state transition and measurement models used by EKF.

## Development environment

A [Nix](https://nixos.org) flake provides the Rust toolchain, Python (with `numpy` and `pytest`) and `maturin`:

```bash
nix develop
cargo test
```

## Python

Build and install the bindings into the current virtualenv with [maturin](https://www.maturin.rs):

```bash
pip install maturin
maturin develop --release      # editable-style install for development
# or
pip install .                  # builds and installs a wheel (needs a Rust toolchain)
```

Then `import determined as dt`.

`dt.ExtendedKalmanFilter` runs the filter maths in Rust and calls back into Python for the model:

```python
ekf = dt.ExtendedKalmanFilter(state, f=f, jac_f=jac_f, q=q, h=h, jac_h=jac_h, r=r)
ekf.predict(dt.Epoch(1))
ekf.update(dt.Observation(z, epoch=dt.Epoch(1)))
```

- **Model functions** are `f(x, t)`, `jac_f(x, t)`, `h(x, t)` and `jac_h(x, t)`. `x` is the state value (a numpy array) and `t` the integer epoch: `f` advances the state from epoch `t` to `t + 1`, and `h` is evaluated at the measurement epoch. They may be time-varying, and may return anything array-like, so Jacobians can come from an autodiff library (e.g. `jax.jacfwd`, which differentiates `x` and not `t`).
- **Keyword-only arguments:** constructors take their first argument positionally and the rest by keyword, e.g. `dt.State(value, covariance=cov, epoch=epoch)`.
- **States are live views.** To avoid copying, the `State` returned by `predict`, `update` and `.state` shares memory with the filter's internal state, so it changes on the next call (and is the same object you passed in). Copy the arrays (`state.value.copy()`) if you need to keep a snapshot.
- **Logging:** messages from the Rust side go through Python's `logging` under the name `determined`, e.g. `logging.getLogger("determined").setLevel(logging.DEBUG)`.

See `examples/ekf.py`. Build a distributable wheel with `maturin build --release`.

## Examples
<img src="docs/assets/TwoMassSpringDamperScene_ManimCE_v0.19.1.gif" alt="spring damper example"/>

Check out examples for `python` bindings and the core `rust` library in `/examples`.  To run `rust` examples use `cargo`:
```bash
cargo run --example <example_name>
```

To run the visualisation example using `python` and `manim` use:
```bash
manim -pql examples/manim_two_mass_spring_damper.py TwoMassSpringDamperScene
```

## Roadmap

- **Gating:** expose the innovation `y = z - h(x)`, its covariance `S` and the normalised innovation squared `yᵀ S⁻¹ y` from `update`, so outlier measurements can be rejected against a chi-squared threshold. Not exposed yet.
- **UD Kalman filter:** a square-root style filter that stores `P = U D Uᵀ` and updates the factors directly, to keep the covariance positive definite by construction. The filters currently use the Joseph form for the covariance update.
- **Automatic Jacobians:** optional `jac_f` / `jac_h` computed by autodiff or finite differences.
