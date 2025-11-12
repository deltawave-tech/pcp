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
      overlays = [
        (final: prev: {
          zigpkgs = inputs.zig-overlay.packages.${prev.system};
          zlspkgs = let orig = zls.packages.${prev.system};
          in orig // {
            zls = orig.zls.overrideAttrs (old: {
              doCheck =
                false; # Disable tests to bypass failures (likely due to emulation timeouts or env issues)
            });
          };
          claude-code = prev.claude-code.overrideAttrs (old: rec {
            version = "2.0.36";
            src = prev.fetchzip {
              url =
                "https://registry.npmjs.org/@anthropic-ai/claude-code/-/claude-code-${version}.tgz";
              hash = "sha256-6tbbCaF1HIgdk1vpbgQnBKWghaKKphGIGZoXtmnhY2I=";
            };
          });
        })
      ];
    in flake-utils.lib.eachSystem [
      "x86_64-linux"
      "aarch64-darwin"
      "x86_64-darwin"
    ] (system:
      let
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };
        lib = pkgs.lib;
        zls = pkgs.zlspkgs.zls;
        pkgsLLVM = if pkgs.stdenv.isLinux then pkgs.pkgsLLVM else pkgs;
        llvmPkg = pkgsLLVM.llvmPackages_git;
        mlirPkg = llvmPkg.mlir.overrideAttrs (old: {
          postInstall = (old.postInstall or "") + ''
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
          # header files. The header files need a couple of '*.h.inc' files.
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
            pkgs.zig
            pkgs.pkg-config
            llvmPkg.lld
            llvmPkg.clang-tools
            llvmPkg.bintools
            llvmPkg.libcxx.dev
            pkgs.zig.hook
          ] ++ lib.optionals pkgs.stdenv.isDarwin [ pkgs.apple-sdk_15 ];
          buildInputs = [
            llvmPkg.libcxx
            llvmPkg.clang-tools
            packages.stablehlo
            llvmPkg.libllvm
            mlirPkg
            pkgs.capnproto
            packages.iree-sdk
          ];
          dontConfigure = true;
          doCheck = true;
          zigBuildFlags = [ "--verbose" "--color" "off" ];
          zigCheckFlags = [ "--verbose" "--color" "off" ];
          zigInstallFlags = [ "--verbose" "--color" "off" ];
          CAPNP_DIR = "${pkgs.capnproto}";
          IREE_SDK_DIR = "${packages.iree-sdk}";
        };
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = packages.pcp.nativeBuildInputs
            ++ [ pkgs.cachix pkgs.claude-code pkgs.lldb pkgs.act zls ];
          buildInputs = packages.pcp.buildInputs;
          shellHook = ''
            echo "Zig development environment loaded"
            echo "Zig version: $(zig version)"
            echo "ZLS version: $(zls --version)"
            export ZIG_GLOBAL_CACHE_DIR="$(pwd)/.zig-cache"
            export CAPNP_DIR="${pkgs.capnproto}"
            export IREE_SDK_DIR="${packages.iree-sdk}"
            ${lib.optionalString pkgs.stdenv.isDarwin ''
              export MACOSX_DEPLOYMENT_TARGET="11.0"
            ''}
          '';
        };
        checks.pcp = packages.pcp;
        packages.iree-sdk = pkgsLLVM.llvmPackages.stdenv.mkDerivation rec {
          pname = "iree-sdk";
          version = "3.8.0";
          src = pkgs.fetchFromGitHub {
            owner = "iree-org";
            repo = "iree";
            rev = "v${version}";
            hash = "sha256-G7fPelraUhiXc/rJSr7J2OAaVSLggvghsSAoaGuGrxc=";
            fetchSubmodules = true;
          };
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.ninja
            pkgs.python3
            pkgsLLVM.llvmPackages.libllvm
          ];
          propagatedBuildInputs = [ pkgsLLVM.llvmPackages.lld ];
          buildInputs = [ pkgs.gtest ];
          cmakeFlags = [
            # Flags suggested in
            # https://iree.dev/building-from-source/getting-started/#configuration-settings
            (lib.cmakeFeature "CMAKE_BUILD_TYPE" "RelWithDebInfo")
            (lib.cmakeFeature "CMAKE_AR" "ar")
            (lib.cmakeFeature "CMAKE_C_COMPILER" "clang")
            (lib.cmakeFeature "CMAKE_CXX_COMPILER" "clang++")

            (lib.cmakeBool "IREE_ENABLE_ASSERTIONS" true)
            (lib.cmakeBool "IREE_ENABLE_SPLIT_DWARF" true)
            (lib.cmakeBool "IREE_ENABLE_THIN_ARCHIVES" true)
            (lib.cmakeBool "IREE_ENABLE_LLD" true)

          ];
          meta = with lib; {
            description = "IREE SDK built from source";
            homepage = "https://iree.dev/";
            platforms = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" ];
          };
        };
      });
}
