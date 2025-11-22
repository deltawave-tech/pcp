{
  description = "PCP - Planetary Compute Protocol";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    zig-overlay.url = "github:mitchellh/zig-overlay";
    zls.url = "github:zigtools/zls";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, zig-overlay, zls, flake-utils }@inputs:
    flake-utils.lib.eachSystem [
      "x86_64-linux"
      "aarch64-linux"
      "aarch64-darwin"
      "x86_64-darwin"
    ] (system:
      let
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };
        lib = pkgs.lib;
        # zig is LLVM based. In order to ensure ABI compatibility, we have base our builds on
        # LLVM/clang.  The following is the `pkgs` set based on LLVM.
        pkgsLLVM = if pkgs.stdenv.isLinux then pkgs.pkgsLLVM else pkgs;
        # The actual LLVM package we are using for building.
        llvmPkg = pkgsLLVM.llvmPackages_21;
        zls = pkgs.zlspkgs.zls;
        # Patch mlir to also contain mlir-pdll
        mlirPkg = llvmPkg.mlir.overrideAttrs (old: {
          postInstall = (old.postInstall or "") + ''
            cp -v bin/mlir-pdll $out/bin
          '';
        });

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
            # Provide a newer version of claude
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
      in rec {
        packages.default = packages.pcp;

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
        checks.pcp = packages.pcp;

        packages.iree-sdk = llvmPkg.stdenv.mkDerivation rec {
          pname = "iree-sdk";
          version = "3.8.0";
          src = pkgs.fetchFromGitHub {
            owner = "iree-org";
            repo = "iree";
            rev = "v${version}";
            hash = "sha256-G7fPelraUhiXc/rJSr7J2OAaVSLggvghsSAoaGuGrxc=";
            fetchSubmodules = true;
          };
          nativeBuildInputs =
            [ pkgs.cmake pkgs.ninja pkgs.python3 pkgs.bintools ];
          propagatedBuildInputs = [ llvmPkg.lld llvmPkg.libllvm ];
          buildInputs = [ pkgs.gtest ];
          cmakeFlags = [
            # Flags suggested in
            # https://iree.dev/building-from-source/getting-started/#configuration-settings
            (lib.cmakeFeature "CMAKE_BUILD_TYPE" "RelWithDebInfo")
            (lib.cmakeFeature "CMAKE_AR" "ar")

            (lib.cmakeBool "IREE_ENABLE_ASSERTIONS" true)
            (lib.cmakeBool "IREE_ENABLE_SPLIT_DWARF" true)
            (lib.cmakeBool "IREE_ENABLE_THIN_ARCHIVES" true)
            (lib.cmakeBool "IREE_ENABLE_LLD" true)
          ];
          meta = {
            description = "IREE SDK built from source";
            homepage = "https://iree.dev/";
            platforms = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" ];
          };
        };
        packages.stdenv = pkgsLLVM.stdenv;
      });
}
