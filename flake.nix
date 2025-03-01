{
  description = "PCP - Planetary Compute Protocol";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
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
      overlays = [
        (final: prev: {
          zigpkgs = inputs.zig-overlay.packages.${prev.system};
          zlspkgs = inputs.zls.packages.${prev.system};
        })
      ];
      systems = builtins.attrNames inputs.zig-overlay.packages;
    in flake-utils.lib.eachSystem systems (system:
      let pkgs = import nixpkgs { inherit overlays system; };
      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            zigpkgs.master
            zlspkgs.zls
            pkgs.glibc
          ];
          shellHook = ''
            echo "Zig development environment loaded"
            echo "Zig version: $(zig version)"
            echo "ZLS version: $(zls --version)"
          '';
        };
      });
}
