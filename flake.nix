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
        zls = pkgs.zls;
        # Patch mlir to also contain mlir-pdll
        mlirPkg = llvmPkg.mlir.overrideAttrs (old: {
          postInstall = (old.postInstall or "") + ''
            cp -v bin/mlir-pdll $out/bin
          '';
        });

        overlays = [
          (final: prev: {
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
          buildInputs = packages.pcp.buildInputs
            ++ packages.pcp.propagatedBuildInputs;
          shellHook = ''
            echo "Zig development environment loaded"
            echo "Zig version: $(zig version)"
            echo "ZLS version: $(zls --version)"
            export ZIG_GLOBAL_CACHE_DIR="$(pwd)/.zig-cache"
            export CAPNP_DIR="${pkgs.capnproto}"
            export IREE_SOURCE_DIR="${packages.iree-sdk.src}"
            export IREE_BUILD_DIR="${packages.iree-sdk.build}"
            ${lib.optionalString pkgs.stdenv.isDarwin ''
              export MACOSX_DEPLOYMENT_TARGET="11.0"
            ''}
          '';
        };

        packages.pcp = llvmPkg.stdenv.mkDerivation {
          name = "pcp";
          version = "main";
          src = ./.;
          # Keep debug symbols/unwind info so Zig safety panics (e.g. integer overflow)
          # produce actionable stack traces in logs (Modal workers are otherwise stripped).
          dontStrip = true;
          nativeBuildInputs = [
            pkgs.makeWrapper
            llvmPkg.libllvm.dev
            mlirPkg.dev
            pkgs.zig_0_13
            pkgs.pkg-config
            llvmPkg.lld
            llvmPkg.clang-tools
            llvmPkg.bintools
            llvmPkg.libcxx.dev
            pkgs.zig_0_13.hook
          ] ++ lib.optionals pkgs.stdenv.isDarwin [ pkgs.apple-sdk_15 ];
          buildInputs = [
            llvmPkg.libcxx
            llvmPkg.clang-tools
            llvmPkg.libllvm
            mlirPkg
            pkgs.capnproto
            packages.iree-sdk.src
            packages.iree-sdk.build
          ];
          propagatedBuildInputs =
            [ pkgs.cudaPackages.cuda_cudart pkgs.glibc packages.iree-sdk pkgs.numactl pkgs.elfutils pkgs.zlib pkgs.zstd pkgs.libdrm ];
          dontConfigure = true;
          doCheck = false;
          zigBuildFlags = [ "--verbose" "--color" "off" ];
          zigCheckFlags = [ "--verbose" "--color" "off" ];
          zigInstallFlags = [ "--verbose" "--color" "off" ];
          CAPNP_DIR = "${pkgs.capnproto}";
          IREE_SOURCE_DIR = "${packages.iree-sdk.src}";
          IREE_BUILD_DIR = "${packages.iree-sdk.build}";

          postFixup = ''
            patchelf --add-needed libnuma.so.1 $out/bin/pcp
            patchelf --add-needed libelf.so.1 $out/bin/pcp
            patchelf --add-needed libz.so.1 $out/bin/pcp
            patchelf --add-needed libzstd.so.1 $out/bin/pcp
            patchelf --add-needed libdrm.so.2 $out/bin/pcp
            patchelf --add-needed libdrm_amdgpu.so.1 $out/bin/pcp

            # Generate the Nix-side RPATH automatically from inputs
            NIX_RPATH="${lib.makeLibraryPath [
              pkgs.numactl
              pkgs.elfutils
              pkgs.zlib
              pkgs.zstd
              pkgs.libdrm
              pkgs.capnproto
              pkgs.cudaPackages.cuda_cudart
              pkgs.glibc
              pkgs.stdenv.cc.cc.lib
              packages.iree-sdk
            ]}"

            # Append System paths for driver fallbacks (PTX JIT, ROCm, etc.)
            SYSTEM_RPATH="/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/lib:/opt/amdgpu/lib/x86_64-linux-gnu:/opt/amdgpu/lib:/run/opengl-driver/lib:/opt/rocm/lib:/opt/rocm/hip/lib"

            patchelf --force-rpath --set-rpath "$NIX_RPATH:$SYSTEM_RPATH" $out/bin/pcp

            # Patch test executables with same RPATH for CUDA/ROCm access
            if [ -f "$out/bin/isolated_vjp_tests" ]; then
              patchelf --force-rpath --set-rpath "$NIX_RPATH:$SYSTEM_RPATH" $out/bin/isolated_vjp_tests
            fi

            wrapProgram $out/bin/pcp \
              --suffix PATH : "${packages.iree-sdk}/bin"
          '';
        };
        checks.pcp = packages.pcp;

        packages.pcp-docker = pkgs.dockerTools.buildImage {
          name = "pcp";
          config = { Cmd = [ "${packages.pcp}/bin/pcp" ]; };
        };

        packages.iree-sdk = llvmPkg.stdenv.mkDerivation rec {
          pname = "iree-sdk";
          version = "3.9.0";
          src = pkgs.fetchFromGitHub {
            owner = "iree-org";
            repo = "iree";
            rev = "v${version}";
            hash = "sha256-O+yp6ysHQJKlgLnoK1esGdRmce6M4nPgFTsxEck0xw0=";
            fetchSubmodules = true;
          };
          nativeBuildInputs =
            [ pkgs.cmake pkgs.ninja pkgs.python3 pkgs.bintools pkgs.patchelf ];
          # Mix together a couple of dependencies needed to build the CUDA layer.  See
          # 'build_tools/scripts/fetch_cuda_deps.sh'.
          postUnpack = ''
            set -e

            echo "Copying needed cuda artifacts"
            icd=/build/iree_cuda_deps
            mkdir -p $icd
            cp -r ${pkgs.cudaPackages_12_2.cuda_cccl.dev}/include $icd
            chmod --recursive u+w $icd
            cp -r ${pkgs.cudaPackages_12_2.cuda_cudart.dev}/include $icd
            chmod --recursive u+w $icd
            cp -r ${pkgs.cudaPackages_12_2.cuda_nvcc}/include $icd
            chmod --recursive u+w $icd

            mkdir -p $icd/nvvm/
            cp -r ${pkgs.cudaPackages_12_2.cuda_nvcc}/nvvm/libdevice/ $icd/nvvm/
            chmod u+w $icd
          '';
          propagatedBuildInputs = [ llvmPkg.lld llvmPkg.libllvm ];
          buildInputs = [ pkgs.gtest ];
          cmakeFlags = [
            # Flags suggested in
            # https://iree.dev/building-from-source/getting-started/#configuration-settings
            (lib.cmakeFeature "CMAKE_BUILD_TYPE" "RelWithDebInfo")
            (lib.cmakeFeature "CMAKE_AR" "ar")
            (lib.cmakeBool "CMAKE_SKIP_INSTALL_RPATH" true)

            (lib.cmakeBool "IREE_ENABLE_ASSERTIONS" true)
            (lib.cmakeBool "IREE_ENABLE_SPLIT_DWARF" true)
            (lib.cmakeBool "IREE_ENABLE_THIN_ARCHIVES" true)
            (lib.cmakeBool "IREE_ENABLE_LLD" true)

            (lib.cmakeFeature "CUDAToolkit_ROOT" "/build/iree_cuda_deps")
            (lib.cmakeFeature "IREE_TARGET_BACKEND_ROCM_DEVICE_BC_PATH"
              "${packages.iree-amdgpu-device-libs}")
            (lib.cmakeBool "IREE_HAL_DRIVER_CUDA" true)
            (lib.cmakeBool "IREE_HAL_DRIVER_HIP" true)
            (lib.cmakeBool "IREE_HAL_DRIVER_VULKAN" true)
            (lib.cmakeBool "IREE_TARGET_BACKEND_CUDA" true)
            (lib.cmakeBool "IREE_TARGET_BACKEND_ROCM" true)
            "-B"
            "/build/build"
          ];
          ninjaFlags = [ "-C" "/build/build" ];

          IREE_CUDA_DEPS_DIR = "/build/iree_cuda_deps";
          preConfigure = ''
            mkdir /build/build
          '';

          outputs = [ "out" "build" ];
          postInstall = ''
            mkdir -p $build
            to_patch=(
              iree-compile
              iree-link
              iree-lld
              iree-mlir-lsp-server
              iree-opt
              iree-reduce
              iree-run-mlir
              test-iree-compiler-api-test-binary
            )

            for f in ''${to_patch[@]}; do
              patchelf --remove-rpath /build/build/tools/$f
            done
            cp -r /build/build/* $build/
          '';

          # Prevent the `_multioutDevs` routine to perform any action. -- It tries to move
          # `$build/lib/cmake` to `$out/lib/cmake` which produces a collision due to the already
          # present `IREE` directory.
          moveToDev = false;
          meta = {
            description = "IREE SDK built from source";
            homepage = "https://iree.dev/";
            platforms = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" ];
          };
        };

        # See ${packages.iree-sdk.src}/compiler/plugins/target/ROCM/CMakeLists.txt
        packages.iree-amdgpu-device-libs = pkgs.fetchzip {
          name = "iree-amdgpu-device-libs";
          url =
            "https://github.com/shark-infra/amdgpu-device-libs/releases/download/v20231101/amdgpu-device-libs-llvm-6086c272a3a59eb0b6b79dcbe00486bf4461856a.tgz";
          hash = "sha256-jCv8pg3oXFVSfvgcSenwxsC/jkyN+dWTwosSAAFEvCo=";
          stripRoot = false;
        };
      });
}
