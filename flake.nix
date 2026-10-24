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
        python = pkgs.python3.withPackages (ps: [
          ps.numpy
          ps.pytest
          # Manim Community, for examples/manim_two_mass_spring_damper.py. The
          # package brings its native dependencies (cairo, pango, ffmpeg...).
          # Its own test suite is skipped: there is no binary cache build on
          # darwin, and one upstream CLI-text test fails when building it.
          (ps.manim.overridePythonAttrs (_: { doCheck = false; }))
        ]);
      in {
        devShells.default = pkgs.mkShell {
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
      });
}
