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
          # A couple of packages need this on darwin.
          config.allowUnsupportedSystem =
            (system == "aarch64-darwin" || system == "x86_64-darwin");
          config.allowBroken =
            (system == "aarch64-darwin" || system == "x86_64-darwin");
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
            ++ packages.pcp.propagatedBuildInputs
            # For wandb_adapter.py
            ++ [ (pkgs.python3.withPackages (ps: [ ps.wandb ])) ];
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
          nativeBuildInputs = [
            llvmPkg.bintools
            llvmPkg.clang-tools
            llvmPkg.libcxx.dev
            llvmPkg.libllvm.dev
            llvmPkg.lld
            mlirPkg.dev
            pkgs.makeWrapper
            pkgs.pkg-config
            pkgs.zig_0_13
            pkgs.zig_0_13.hook
          ] ++ lib.optionals pkgs.stdenv.isDarwin [
            pkgs.apple-sdk_15
          ];
          buildInputs = [
            llvmPkg.clang-tools
            llvmPkg.libcxx
            llvmPkg.libllvm
            mlirPkg
            packages.iree-sdk.build
            packages.iree-sdk.src
            pkgs.capnproto
          ] ++ lib.optionals pkgs.stdenv.isDarwin [
            pkgs.darwin.apple_sdk.frameworks.Foundation
            pkgs.darwin.apple_sdk.frameworks.Metal
            pkgs.darwin.apple_sdk.frameworks.CoreGraphics
          ];
          propagatedBuildInputs = [
            packages.iree-sdk
            pkgs.zlib
            pkgs.zstd
          ] ++ lib.optionals pkgs.stdenv.isLinux [
            pkgs.cudaPackages.cuda_cudart
            pkgs.elfutils
            pkgs.glibc
            pkgs.libdrm
            pkgs.numactl
          ];
          env.ZIG_SYSTEM_LINKER_HACK = "1";
          dontConfigure = true;
          doCheck = false;
          zigBuildFlags = [ "--verbose" "--color" "off" ];
          zigCheckFlags = [ "--verbose" "--color" "off" ];
          zigInstallFlags = [ "--verbose" "--color" "off" ];
          CAPNP_DIR = "${pkgs.capnproto}";
          IREE_SOURCE_DIR = "${packages.iree-sdk.src}";
          IREE_BUILD_DIR = "${packages.iree-sdk.build}";

          postFixup = if pkgs.stdenv.isDarwin then ''
            wrapProgram $out/bin/pcp \
              --suffix PATH : "${packages.iree-sdk}/bin"
          '' else ''
            patchelf --add-needed libnuma.so.1 $out/bin/pcp
            patchelf --add-needed libelf.so.1 $out/bin/pcp
            patchelf --add-needed libz.so.1 $out/bin/pcp
            patchelf --add-needed libzstd.so.1 $out/bin/pcp
            patchelf --add-needed libdrm.so.2 $out/bin/pcp
            patchelf --add-needed libdrm_amdgpu.so.1 $out/bin/pcp

            NIX_RPATH="${
              lib.makeLibraryPath [
                packages.iree-sdk
                pkgs.capnproto
                pkgs.cudaPackages.cuda_cudart
                pkgs.elfutils
                pkgs.glibc
                pkgs.libdrm
                pkgs.numactl
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
                pkgs.zstd
              ]
            }"

            SYSTEM_RPATH="/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/lib:/opt/amdgpu/lib/x86_64-linux-gnu:/opt/amdgpu/lib:/run/opengl-driver/lib:/opt/rocm/lib:/opt/rocm/hip/lib"

            patchelf --force-rpath --set-rpath "$NIX_RPATH:$SYSTEM_RPATH" $out/bin/pcp

            if [ -f "$out/bin/isolated_vjp_tests" ]; then
              patchelf --force-rpath --set-rpath "$NIX_RPATH:$SYSTEM_RPATH" $out/bin/isolated_vjp_tests
            fi

            wrapProgram $out/bin/pcp \
              --prefix PATH : "${packages.pcp-wandb-adapter}/bin" \
              --suffix PATH : "${packages.iree-sdk}/bin"
          '';
        };
        checks.pcp = packages.pcp;

        packages.pcp-wandb-adapter = let
          wandb_module_splitted = pkgs.lib.strings.splitString "\n"
            (builtins.readFile tools/wandb_adapter.py);
          wandb_module_cut_shebang =
            pkgs.lib.lists.drop 1 wandb_module_splitted;
          wandb_module_no_shebang =
            builtins.concatStringsSep "\n" wandb_module_cut_shebang;
        in pkgs.writers.writePython3Bin "wandb_adapter.py" {
          libraries = [ pkgs.python3Packages.wandb ];
        } wandb_module_no_shebang;

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

          nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.python3 pkgs.bintools ]
            ++ lib.optionals pkgs.stdenv.isLinux [ pkgs.patchelf ]
            ++ lib.optionals pkgs.stdenv.isDarwin [ pkgs.apple-sdk_15 ];

          postUnpack = lib.optionalString pkgs.stdenv.isLinux ''
            set -e
            echo "Copying needed cuda artifacts"
            icd=$NIX_BUILD_TOP/iree_cuda_deps
            mkdir -p $icd
            cp -r ${pkgs.cudaPackages_12_2.cuda_cccl.dev}/include $icd
            chmod -R u+w $icd
            cp -r ${pkgs.cudaPackages_12_2.cuda_cudart.dev}/include $icd
            chmod -R u+w $icd
            cp -r ${pkgs.cudaPackages_12_2.cuda_nvcc}/include $icd
            chmod -R u+w $icd
            mkdir -p $icd/nvvm/
            cp -r ${pkgs.cudaPackages_12_2.cuda_nvcc}/nvvm/libdevice/ $icd/nvvm/
            chmod u+w $icd
          '';

          propagatedBuildInputs = [ llvmPkg.lld llvmPkg.libllvm ];

          buildInputs = [ pkgs.gtest ] ++ lib.optionals pkgs.stdenv.isDarwin [
            pkgs.darwin.apple_sdk.frameworks.Foundation
            pkgs.darwin.apple_sdk.frameworks.Metal
          ];

          cmakeFlags = [
            (lib.cmakeFeature "CMAKE_BUILD_TYPE" "RelWithDebInfo")
            (lib.cmakeFeature "CMAKE_AR" "ar")
            (lib.cmakeBool "IREE_BUILD_TESTS" false)
            (lib.cmakeBool "IREE_BUILD_SAMPLES" false)
            (lib.cmakeBool "CMAKE_SKIP_INSTALL_RPATH" true)
            (lib.cmakeBool "IREE_ENABLE_ASSERTIONS" true)
            (lib.cmakeBool "IREE_ENABLE_SPLIT_DWARF" true)
            (lib.cmakeBool "IREE_ENABLE_THIN_ARCHIVES" pkgs.stdenv.isLinux)
            (lib.cmakeBool "IREE_ENABLE_LLD" true)
            (lib.cmakeBool "IREE_TARGET_BACKEND_METAL_SPIRV" true)
            (lib.cmakeBool "IREE_HAL_DRIVER_METAL" pkgs.stdenv.isDarwin)
            (lib.cmakeBool "IREE_HAL_DRIVER_VULKAN" true)
            "-B" "build"
          ] ++ lib.optionals pkgs.stdenv.isLinux [
            (lib.cmakeBool "IREE_TARGET_BACKEND_ROCM" true)
            (lib.cmakeBool "IREE_HAL_DRIVER_HIP" true)
            (lib.cmakeBool "IREE_HAL_DRIVER_CUDA" true)
            (lib.cmakeBool "IREE_TARGET_BACKEND_CUDA" true)
            (lib.cmakeFeature "CUDAToolkit_ROOT" "../iree_cuda_deps")
            (lib.cmakeFeature "IREE_TARGET_BACKEND_ROCM_DEVICE_BC_PATH"
              "${packages.iree-amdgpu-device-libs}")
          ];

          ninjaFlags = [ "-C" "build" ];

          preConfigure = ''
            mkdir -p build
          '' + lib.optionalString pkgs.stdenv.isDarwin ''
            export NIX_CFLAGS_COMPILE="$NIX_CFLAGS_COMPILE \
              -DMTLLanguageVersion3_0=196608 \
              -DMTLGPUFamilyApple8=1008 \
              -DMTLGPUFamilyMetal3=5001"
          '';

          outputs = [ "out" "build" ];

          postInstall = ''
            mkdir -p $build
            cp -r build/* $build/
          '' + lib.optionalString pkgs.stdenv.isLinux ''
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
              if [ -f "$build/tools/$f" ]; then
                patchelf --remove-rpath $build/tools/$f
              fi
            done
          '';

          moveToDev = false;
          meta = {
            description = "IREE SDK built from source";
            homepage = "https://iree.dev/";
            platforms = [
              "aarch64-darwin"
              "aarch64-linux"
              "x86_64-darwin"
              "x86_64-linux"
            ];
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
