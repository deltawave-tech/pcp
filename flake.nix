{
  description = "PCP - Planetary Compute Protocol";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    zig-overlay = {
      url = "github:mitchellh/zig-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
    zls = {
      url = "github:zigtools/zls";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.zig-overlay.follows = "zig-overlay";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, zig-overlay, zls, flake-utils }@inputs:
    let
      # Inject zig-overlay and zls into the package tree
      overlays = [
        (final: prev: {
          zigpkgs = inputs.zig-overlay.packages.${prev.system};
          zlspkgs = inputs.zls.packages.${prev.system};
        })
      ];
      systems = builtins.attrNames inputs.zig-overlay.packages;
    in flake-utils.lib.eachSystem systems (system:
      let
        pkgs = import nixpkgs { inherit overlays system; };
        zig = zig-overlay.packages.${system}.master;
      in rec {
        packages.default = packages.pcp;
        packages.pcp = pkgs.stdenvNoCC.mkDerivation {
          name = "pcp";
          version = "main";
          src = ./.;
          nativeBuildInputs = [ zig ];
          dontConfigure = true;
          dontInstall = true;
          doCheck = true;
          buildPhase = ''
            NO_COLOR=1 # prevent escape codes from messing up the `nix log`
            zig build install --global-cache-dir $(pwd)/.cache -Dcpu=baseline -Doptimize=ReleaseSafe --prefix $out
          '';
          checkPhase = let
            targets = [
              "test"
              "run"
              "run-autodiff-test"
              "run-comptime-examples"
              "run-plan-test"
            ];
            buildCmd = target:
              "zig build ${target} --global-cache-dir $(pwd)/.cache -Dcpu=baseline";
          in builtins.concatStringsSep "\n" (map buildCmd targets);
        };
        # The development environment draws in the Zig compiler and ZLS.
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [ lldb zig zlspkgs.zls ];
          shellHook = ''
            echo "Zig development environment loaded"
            echo "Zig version: $(zig version)"
            echo "ZLS version: $(zls --version)"
          '';
        };
      });
}
