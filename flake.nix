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
          claude-code = prev.claude-code.overrideAttrs (old: rec {
            version = "1.0.85";
            src = prev.fetchzip {
              url =
                "https://registry.npmjs.org/@anthropic-ai/claude-code/-/claude-code-${version}.tgz";
              hash = "sha256-CLqvcolG94JBC5VFlsfybZ9OXe81gJBzKU6Xgr7CGWo=";

            };
          });
        })
      ];
      systems = builtins.attrNames inputs.zig-overlay.packages;
    in flake-utils.lib.eachSystem systems (system:
      let
        pkgs = import nixpkgs {
          inherit overlays system;
          config.allowUnfree = true;
        };
        lib = pkgs.lib;

        # We use zig from nixpkgs. Alternatively:
        #
        zig = zig-overlay.packages.${system}."0.14.0";
        #
        # zig = zig-overlay.packages.${system}.master;
        # zig = pkgs.zig;
        # zls from nixpkgs lines up nicely with the zig "default" compiler in nixpkgs. If working on
        # bleeding edge:
        #
        # zls = pkgs.zlspkgs.zls;
        zls = pkgs.zls;

        # Shortcuts for llvm -- as we have to link from Zig (clang) to LLVM we need an LLVM which is
        # itself based on LLVM. This is the default on Darwin/MacOS. On Linux we have to use
        # `pkgs.pkgsLLVM` instead. Sadly, this involves some compilation. 
        llvmPkg = (if pkgs.stdenv.isLinux then
          pkgs.pkgsLLVM
        else
          pkgs).llvmPackages_git;
        # The default 'mlir' package does not expose `mlir-pdll`
        mlirPkg = llvmPkg.mlir.overrideAttrs (old: {
          postInstall = (old.postInstall or "") + ''
            echo "Installing mlir-pdll"
            cp -v bin/mlir-pdll $out/bin
          '';
        });

      in rec {
        packages.default = packages.pcp;

        packages.stablehlo = let
          libllvm = llvmPkg.libllvm;
          tblgen = llvmPkg.tblgen;
        in llvmPkg.stdenv.mkDerivation rec {
          name = "stablehlo";
          version = "1.13.0";
          src = pkgs.fetchFromGitHub {
            owner = "openxla";
            repo = "stablehlo";
            rev = "v${version}";
            hash = "sha256-OH6Nz4b5sR8sPII59p6AaHXY7VlUS3pJM5ejdrri4iw=";
          };
          outputs = [ "out" "dev" ];
          nativeBuildInputs = [ mlirPkg.dev tblgen ]
            ++ (with pkgs; [ cmake gtest ninja pkg-config ]);
          buildInputs = [ mlirPkg libllvm ] ++ (with pkgs; [ libffi libxml2 ]);
          cmakeFlags = [
            (lib.cmakeBool "LLVM_ENABLE_LDD" true)
            (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
            (lib.cmakeFeature "STABLEHLO_ENABLE_BINDINGS_PYTHON" "OFF")
            (lib.cmakeFeature "MLIR_DIR" "${mlirPkg.dev}/lib/cmake/mlir")
            (lib.cmakeFeature "LLVM_DIR" "${libllvm.dev}/lib/cmake/llvm")
          ];
          buildPhase = ''
            cd $TMPDIR
            cmake \
              $src \
              -GNinja \
              $cmakeFlags

            # ninja: error: 'stablehlo/reference/mlir-tblgen', needed by 'stablehlo/reference/InterpreterOps.h.inc', missing and no known rule to make it
            ln -s ${tblgen}/bin/mlir-tblgen stablehlo/dialect/
            ln -s ${tblgen}/bin/mlir-tblgen stablehlo/reference/
            ln -s ${tblgen}/bin/mlir-tblgen stablehlo/tests/
            ln -s ${tblgen}/bin/mlir-tblgen stablehlo/conversions/linalg/transforms/
            ln -s ${tblgen}/bin/mlir-tblgen stablehlo/conversions/tosa/transforms/
            ln -s ${mlirPkg}/bin/mlir-pdll stablehlo/conversions/tosa/transforms/
            ln -s ${tblgen}/bin/mlir-tblgen stablehlo/transforms/
            ln -s ${tblgen}/bin/mlir-tblgen stablehlo/transforms/optimization/
            cmake --build . --verbose
          '';
          # See https://github.com/openxla/stablehlo/issues/2811 -- StableHLO does not install
          # header files.  The header files need a couple of '*.h.inc' files.
          postInstall = ''
            mkdir -p $dev/include/
            cd /build/source/
            find stablehlo/ -name '*.h' -exec cp -v --parents {} $dev/include/ \;
            cd /build
            find stablehlo/ -name '*.h.inc' -exec cp -v --parents {} $dev/include/ \;
          '';
        };
        packages.pcp = llvmPkg.stdenv.mkDerivation {
          name = "pcp";
          version = "main";
          src = ./.;
          nativeBuildInputs = [
            llvmPkg.libllvm.dev
            mlirPkg.dev
            zig
            packages.stablehlo.dev
            pkgs.pkg-config
            pkgs.spirv-cross
            llvmPkg.lld
            llvmPkg.clang-tools
            llvmPkg.bintools
          ];
          buildInputs = [
            llvmPkg.libcxx.dev
            llvmPkg.libcxx
            llvmPkg.clang-tools
            packages.stablehlo
            llvmPkg.libllvm
            llvmPkg.mlir-tools
            mlirPkg
            pkgs.spirv-cross
            pkgs.capnproto
          ];

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
          nativeBuildInputs = packages.pcp.nativeBuildInputs ++ (with pkgs; [
            # claude-code is taken from the overlay defined above
            claude-code
            lldb
            shfmt
          ]) ++ [ zig zls ];
          shellHook = ''
            echo "Zig development environment loaded"
            echo "Zig version: $(zig version)"
            echo "ZLS version: $(zls --version)"
            echo "${packages.stablehlo.dev}"
            export CAPNP_DIR="${pkgs.capnproto}"
            echo "libcxx=${llvmPkg.libcxx}"
            echo "cc=${pkgs.stdenv.cc.cc}"
          '';
        };
        checks.pcp = packages.pcp;
      });
}
