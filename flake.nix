{
  description = "PCP - Planetary Compute Protocol";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    zig-overlay.url = "github:mitchellh/zig-overlay";
    zls.url = "github:zigtools/zls";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, zig-overlay, zls, flake-utils }@inputs:
    let
      # Your original, correct overlay structure
      overlays = [
        (final: prev: {
          zigpkgs = inputs.zig-overlay.packages.${prev.system};
          zlspkgs = zls.packages.${prev.system};
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
    in
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };
        lib = pkgs.lib;
        zls = pkgs.zlspkgs.zls;

        pkgsLLVM = if pkgs.stdenv.isLinux then pkgs.pkgsLLVM else pkgs;

        llvmPkg = pkgsLLVM.llvmPackages_21;

        mlirPkg = llvmPkg.mlir.overrideAttrs (old: {
          postInstall = (old.postInstall or "") + ''
            cp -v bin/mlir-pdll $out/bin
          '';
        });

        # IREE SDK implementation (This is correct)
        iree-sdk-srcs = {
          x86_64-linux = {
            url = "https://github.com/openxla/iree/releases/download/20240415.156/iree-sdk-linux-x86_64-20240415.156.tar.gz";
            hash = "sha256-Wl3Yg4xZtOI0nRQjwSDy4y6Lq89L2g8KjB/9a/m8v9o=";
          };

          # --- VERIFIED aarch64-darwin CONFIGURATION ---
          aarch64-darwin = {
            url = "https://github.com/openxla/iree/releases/download/20240415.156/iree-sdk-macos-arm64-nightly-20240415.156.tar.gz";
            hash = "sha256-o12M2q85jF23/E2s6v9KMOI8pL7VpY7FkC/k4s0D6H4=";
          };
          # ----------------------------------------------

          x86_64-darwin = {
            # Note: This URL might also need '-nightly' if you use it.
            url = "https://github.com/openxla/iree/releases/download/20240415.156/iree-sdk-macos-x86_64-20240415.156.tar.gz";
            hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="; # Placeholder
          };
        };

        iree-sdk = pkgs.stdenv.mkDerivation {
          pname = "iree-sdk";
          version = "20240415.156";
          src = pkgs.fetchurl {
            url = iree-sdk-srcs.${system}.url;
            hash = iree-sdk-srcs.${system}.hash;
          };
          installPhase = ''
            mkdir -p $out
            cp -r tools $out/tools; cp -r lib $out/lib; cp -r include $out/include
            chmod +x $out/tools/*
          '';
          meta = with lib; {
            description = "IREE SDK";
            homepage = "https://iree.dev/";
            platforms = builtins.attrNames iree-sdk-srcs;
          };
        };

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
            env | sort
            mkdir -p $dev/include/
            find stablehlo/ -name '*.h.inc' -exec cp -v --parents {} $dev/include/ \;

            cd $src
            find stablehlo/ -name '*.h' -exec cp -v --parents {} $dev/include/ \;
          '';
        };

        packages.pcp = llvmPkg.stdenv.mkDerivation {
          name = "pcp";
          version = "main";
          src = ./.;
          nativeBuildInputs = [
            llvmPkg.libllvm.dev
            mlirPkg.dev
            packages.stablehlo.dev
            iree-sdk
            pkgs.zig
            pkgs.pkg-config
            llvmPkg.lld
            llvmPkg.clang-tools
            llvmPkg.bintools
            llvmPkg.libcxx.dev
            pkgs.zig.hook
          ] ++ lib.optionals pkgs.stdenv.isDarwin [
            pkgs.apple-sdk_15
          ];
          buildInputs = [
            llvmPkg.libcxx
            llvmPkg.clang-tools
            packages.stablehlo
            llvmPkg.libllvm
            mlirPkg
            pkgs.capnproto
            iree-sdk
          ];

          dontConfigure = true;
          doCheck = true;
          zigBuildFlags = [ "--verbose" "--color" "off" ];
          zigCheckFlags = [ "--verbose" "--color" "off" ];
          zigInstallFlags = [ "--verbose" "--color" "off" ];

          CAPNP_DIR = "${pkgs.capnproto}";
          IREE_SDK_DIR = "${iree-sdk}";
        };

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = packages.pcp.nativeBuildInputs ++ [
            pkgs.cachix
            pkgs.claude-code
            pkgs.lldb
            pkgs.act
            zls
          ];
          buildInputs = packages.pcp.buildInputs;
          shellHook = ''
            echo "Zig development environment loaded"
            echo "Zig version: $(zig version)"
            echo "ZLS version: $(zls --version)"

            export ZIG_GLOBAL_CACHE_DIR="$(pwd)/.zig-cache"
            export CAPNP_DIR="${pkgs.capnproto}"
            export IREE_SDK_DIR="${iree-sdk}"

            ${lib.optionalString pkgs.stdenv.isDarwin ''
              export MACOSX_DEPLOYMENT_TARGET="11.0"
            ''}
          '';
        };

        checks.pcp = packages.pcp;
      });
}
