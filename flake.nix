{
  description = "determined: Rust filtering library with Python bindings";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        pythonPackages = ps: [ ps.numpy ps.pytest ];
        python = pkgs.python3.withPackages (ps: pythonPackages ps ++ [
          # Manim Community, for examples/manim_two_mass_spring_damper.py. The
          # package brings its native dependencies (cairo, pango, ffmpeg...).
          # Its own test suite is skipped: there is no binary cache build on
          # darwin, and one upstream CLI-text test fails when building it.
          (ps.manim.overridePythonAttrs (_: { doCheck = false; }))
        ]);

        shell = python: pkgs.mkShell {
          packages = with pkgs; [
            # Rust
            cargo
            rustc
            clippy
            rustfmt
            rust-analyzer

            # Python bindings
            python
            maturin
          ];

          env.RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
        };
      in {
        devShells.default = shell python;

        # CI: the same without manim, which would otherwise be built from
        # source on every run (the override above changes its hash)
        devShells.ci = shell (pkgs.python3.withPackages pythonPackages);
      });
}
