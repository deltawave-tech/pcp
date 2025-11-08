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
        # zig = zig-overlay.packages.${system}."0.14.0";
        #
        # zig = zig-overlay.packages.${system}.master;
        zig = pkgs.zig;
        # zls from nixpkgs lines up nicely with the zig "default" compiler in nixpkgs. If working on
        # bleeding edge:
        #
        # zls = pkgs.zlspkgs.zls;
        zls = pkgs.zls;

        # Shortcuts for llvm -- as we have to link from Zig (clang) to LLVM we need an LLVM which is
        # itself based on LLVM. This is the default on Darwin/MacOS. On Linux we have to use
        # `pkgs.pkgsLLVM` instead. Sadly, this involves some compilation.
        pkgsLLVM = if pkgs.stdenv.isLinux then pkgs.pkgsLLVM else pkgs;
        llvmPkg = pkgsLLVM.llvmPackages_git;
        # The default 'mlir' package does not expose `mlir-pdll`
        mlirPkg = llvmPkg.mlir.overrideAttrs (old: {
          postInstall = (old.postInstall or "") + ''
            echo "Installing mlir-pdll"
            cp -v bin/mlir-pdll $out/bin
          '';
        });


        # NEW: IREE SDK from pre-built binaries (with CORRECTED URLs and naming)
        iree-sdk-srcs = {
          # For Linux
          x86_64-linux = {
            url = "https://github.com/openxla/iree/releases/download/20240415.156/iree-sdk-linux-x86_64-20240415.156.tar.gz";
            hash = "sha256-Wl3Yg4xZtOI0nRQjwSDy4y6Lq89L2g8KjB/9a/m8v9o=";
          };
          # For Apple Silicon
          aarch64-darwin = {
            url = "https://github.com/openxla/iree/releases/download/20240415.156/iree-sdk-macos-arm64-20240415.156.tar.gz";
            hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="; #<-- Placeholder, see below
          };
          # For Intel Mac
          x86_64-darwin = {
            url = "https://github.com/openxla/iree/releases/download/20240415.156/iree-sdk-macos-x86_64-20240415.156.tar.gz";
            hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="; #<-- Placeholder, see below
          };
        };

        # Create the iree-sdk package
        iree-sdk = pkgs.stdenv.mkDerivation {
          pname = "iree-sdk";
          version = "20240415.156"; # Match the release version

          src = pkgs.fetchurl {
            url = iree-sdk-srcs.${system}.url;
            hash = iree-sdk-srcs.${system}.hash;
          };
          
          # This phase is now simpler as the new archives have a clear structure
          installPhase = ''
            mkdir -p $out
            cp -r tools $out/tools
            cp -r lib $out/lib
            cp -r include $out/include
            
            # Make sure binaries in tools are executable
            chmod +x $out/tools/iree-compile
            chmod +x $out/tools/iree-run-module
          '';
          
          # Add metadata
          meta = with lib; {
            description = "IREE (Intermediate Representation Execution Environment) SDK";
            homepage = "https://iree.dev/";
            platforms = [ "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ];
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
            iree-sdk # <-- ADD THE SDK HERE
            zig
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
            pkgsLLVM.capnproto
            iree-sdk # <-- AND HERE

            llvmPkg.libllvm.dev
            mlirPkg.dev
            packages.stablehlo.dev
            zig
            pkgs.pkg-config
            llvmPkg.lld
            llvmPkg.clang-tools
            llvmPkg.bintools
            llvmPkg.libcxx.dev

          ];

          dontConfigure = true;
          doCheck = true;
          zigBuildFlags = [ "--verbose" "--color" "off" ];
          zigCheckFlags = [ "--verbose" "--color" "off" ];
          zigInstallFlags = [ "--verbose" "--color" "off" ];

          # TODO properly integrate this via pkg-config
          CAPNP_DIR = "${pkgs.capnproto}";
          
          # NEW: Pass the path to the SDK as an environment variable
          # so `build.zig` can find it easily.
          IREE_SDK_DIR = "${iree-sdk}";
        };
        # The development environment draws in the Zig compiler and ZLS.
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = packages.pcp.nativeBuildInputs ++ (with pkgs; [
            cachix
            # claude-code is taken from the overlay defined above
            claude-code
            lldb
            # act is for local work on GitHub Actions.  We also use it to settle cachix.
            act
          ]) ++ [ zls ];
          buildInputs = packages.pcp.buildInputs;
          shellHook = ''
            echo "Zig development environment loaded"
            echo "Zig version: $(zig version)"
            echo "ZLS version: $(zls --version)"

            ZIG_GLOBAL_CACHE_DIR="$(pwd)/.zig-cache"
            export ZIG_GLOBAL_CACHE_DIR

            CAPNP_DIR="${pkgs.capnproto}"
            export CAPNP_DIR

            # Also export it in the development shell for local builds
            export IREE_SDK_DIR="${iree-sdk}"

            ${lib.optionalString pkgs.stdenv.isDarwin ''
              export MACOSX_DEPLOYMENT_TARGET="11.0"
            ''}
          '';
        };
        checks.pcp = packages.pcp;
      });
}
