const std = @import("std");
const builtin = @import("builtin");

const test_targets = [_]std.Target.Query{
    .{}, // native
    .{ .cpu_arch = .aarch64, .os_tag = .macos },
    .{ .cpu_arch = .aarch64, .os_tag = .linux },
    .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu },
    .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .musl },
    .{ .cpu_arch = .x86_64, .os_tag = .windows },
};

// MLIR configuration
const MLIRConfig = struct {
    enabled: bool = false,
    llvm_config_path: ?[]const u8 = null,
    lib_dir: ?[]const u8 = null,
    include_dir: ?[]const u8 = null,
};

// Cap'n Proto configuration
const CapnpConfig = struct {
    enabled: bool = false,
    include_dir: ?[]const u8 = null,
    lib_dir: ?[]const u8 = null,
};

// Attempt to find LLVM/MLIR installation
fn detectMLIR(b: *std.Build) MLIRConfig {
    var config = MLIRConfig{};

    // 1. Prioritize an environment variable for the LLVM build directory
    if (std.process.getEnvVarOwned(b.allocator, "LLVM_DIR")) |llvm_dir| {
        defer b.allocator.free(llvm_dir);
        const llvm_config_path = std.fs.path.join(b.allocator, &[_][]const u8{ llvm_dir, "bin", "llvm-config" }) catch {
            std.debug.print("Failed to construct llvm-config path from LLVM_DIR\n", .{});
            return config;
        };
        defer b.allocator.free(llvm_config_path);
        
        // Test if this llvm-config works
        const result = std.process.Child.run(.{
            .allocator = b.allocator,
            .argv = &[_][]const u8{ llvm_config_path, "--version" },
        }) catch |err| {
            std.debug.print("LLVM_DIR llvm-config test failed: {}\n", .{err});
            return config;
        };
        defer b.allocator.free(result.stdout);
        defer b.allocator.free(result.stderr);
        
        if (result.term == .Exited and result.term.Exited == 0) {
            config.enabled = true;
            config.llvm_config_path = b.dupe(llvm_config_path);
            std.debug.print("Found LLVM via LLVM_DIR at: {s}\n", .{llvm_config_path});
            return config;
        }
    } else |_| {
        // Environment variable not set, continue with auto-detection
    }

    // 2. If no environment variable, proceed with auto-detection
    const llvm_config_candidates = [_][]const u8{
        // Local build paths (relative to project root)
        "llvm-build/bin/llvm-config",
        "./llvm-build/bin/llvm-config",
        // System llvm-config
        "llvm-config",
        "llvm-config-18",
        "llvm-config-17",
        "llvm-config-16",
        // Common system installation paths
        "/usr/local/bin/llvm-config",
        "/opt/homebrew/bin/llvm-config",
        "/opt/homebrew/opt/llvm/bin/llvm-config", // Homebrew keg-only install
        "/usr/bin/llvm-config",
    };
    
    for (llvm_config_candidates) |candidate| {
        const result = std.process.Child.run(.{
            .allocator = b.allocator,
            .argv = &[_][]const u8{ candidate, "--version" },
        }) catch continue;
        defer b.allocator.free(result.stdout);
        defer b.allocator.free(result.stderr);
        
        if (result.term == .Exited and result.term.Exited == 0) {
            config.enabled = true;
            config.llvm_config_path = b.dupe(candidate);
            std.debug.print("Found LLVM at: {s}\n", .{candidate});
            break;
        }
    }
    
    if (config.enabled and config.llvm_config_path != null) {
        // Get library directory
        const lib_result = std.process.Child.run(.{
            .allocator = b.allocator,
            .argv = &[_][]const u8{ config.llvm_config_path.?, "--libdir" },
        }) catch {
            config.enabled = false;
            return config;
        };
        defer b.allocator.free(lib_result.stdout);
        defer b.allocator.free(lib_result.stderr);
        
        if (lib_result.term == .Exited and lib_result.term.Exited == 0) {
            const lib_dir = std.mem.trim(u8, lib_result.stdout, " \n\r\t");
            config.lib_dir = b.dupe(lib_dir);
        }
        
        // Get include directory
        const inc_result = std.process.Child.run(.{
            .allocator = b.allocator,
            .argv = &[_][]const u8{ config.llvm_config_path.?, "--includedir" },
        }) catch {
            config.enabled = false;
            return config;
        };
        defer b.allocator.free(inc_result.stdout);
        defer b.allocator.free(inc_result.stderr);
        
        if (inc_result.term == .Exited and inc_result.term.Exited == 0) {
            const inc_dir = std.mem.trim(u8, inc_result.stdout, " \n\r\t");
            config.include_dir = b.dupe(inc_dir);
        }
        
        std.debug.print("MLIR Configuration:\n", .{});
        std.debug.print("  Library directory: {s}\n", .{config.lib_dir orelse "unknown"});
        std.debug.print("  Include directory: {s}\n", .{config.include_dir orelse "unknown"});
    } else {
        std.debug.print("LLVM/MLIR not found. MLIR features will be disabled.\n", .{});
        std.debug.print("To enable MLIR support, install LLVM with MLIR enabled:\n", .{});
        std.debug.print("  macOS: brew install llvm\n", .{});
        std.debug.print("  Ubuntu: apt install llvm-dev libmlir-dev\n", .{});
        std.debug.print("  Or build from source with -DLLVM_ENABLE_PROJECTS=mlir\n", .{});
    }
    
    return config;
}

// Attempt to find Cap'n Proto installation
fn detectCapnp(b: *std.Build) CapnpConfig {
    var config = CapnpConfig{};

    // 1. Prioritize an environment variable for the Cap'n Proto directory
    if (std.process.getEnvVarOwned(b.allocator, "CAPNP_DIR")) |capnp_dir| {
        defer b.allocator.free(capnp_dir);
        
        const include_dir = std.fs.path.join(b.allocator, &[_][]const u8{ capnp_dir, "include" }) catch {
            std.debug.print("Failed to construct include path from CAPNP_DIR\n", .{});
            return config;
        };
        defer b.allocator.free(include_dir);
        
        const lib_dir = std.fs.path.join(b.allocator, &[_][]const u8{ capnp_dir, "lib" }) catch {
            std.debug.print("Failed to construct lib path from CAPNP_DIR\n", .{});
            return config;
        };
        defer b.allocator.free(lib_dir);
        
        // Test if directories exist
        if (std.fs.cwd().access(include_dir, .{})) |_| {
            if (std.fs.cwd().access(lib_dir, .{})) |_| {
                config.enabled = true;
                config.include_dir = b.dupe(include_dir);
                config.lib_dir = b.dupe(lib_dir);
                std.debug.print("Found Cap'n Proto via CAPNP_DIR at: {s}\n", .{capnp_dir});
                return config;
            } else |_| {
                std.debug.print("CAPNP_DIR lib directory not accessible, continuing with auto-detection\n", .{});
            }
        } else |_| {
            std.debug.print("CAPNP_DIR include directory not accessible, continuing with auto-detection\n", .{});
        }
    } else |_| {
        // Environment variable not set, continue with auto-detection
    }

    // 2. If no environment variable, proceed with auto-detection of common paths
    const capnp_candidates = [_]struct { include: []const u8, lib: []const u8 }{
        // Homebrew paths (macOS)
        .{ .include = "/opt/homebrew/opt/capnp/include", .lib = "/opt/homebrew/lib" },
        .{ .include = "/opt/homebrew/include", .lib = "/opt/homebrew/lib" },
        // Standard Linux/Unix paths
        .{ .include = "/usr/local/include", .lib = "/usr/local/lib" },
        .{ .include = "/usr/include", .lib = "/usr/lib" },
        // Alternative system paths
        .{ .include = "/usr/local/opt/capnp/include", .lib = "/usr/local/lib" },
    };
    
    for (capnp_candidates) |candidate| {
        // Check if capnp headers exist (look for common.h which should always be present)
        const header_path = std.fs.path.join(b.allocator, &[_][]const u8{ candidate.include, "capnp", "common.h" }) catch continue;
        defer b.allocator.free(header_path);
        
        if (std.fs.cwd().access(header_path, .{})) |_| {
            config.enabled = true;
            config.include_dir = b.dupe(candidate.include);
            config.lib_dir = b.dupe(candidate.lib);
            std.debug.print("Found Cap'n Proto at: include={s}, lib={s}\n", .{ candidate.include, candidate.lib });
            break;
        } else |_| {
            // Continue checking other candidates
        }
    }
    
    if (!config.enabled) {
        std.debug.print("Cap'n Proto not found in standard locations\n", .{});
    }
    
    return config;
}

// Attempt to find IREE SDK installation
fn detectIree(b: *std.Build) ?[]const u8 {
    return std.process.getEnvVarOwned(b.allocator, "IREE_SDK_DIR") catch null;
}

// Helper function to execute a command and capture its output
fn getCommandOutput(b: *std.Build, argv: []const []const u8) ![]const u8 {
    const result = std.process.Child.run(.{
        .allocator = b.allocator,
        .argv = argv,
    }) catch {
        std.debug.print("Failed to run {s}\n", .{argv});
        return error.CmdFailed;
    };
    
    if (result.term != .Exited or result.term.Exited != 0) {
        std.debug.print("Command {s} failed with exit code {}\n", .{ argv, result.term });
        return error.CmdFailed;
    }
    
    return b.allocator.dupe(u8, std.mem.trim(u8, result.stdout, " \n\r\t"));
}

// Enhanced MLIR support function with consolidated library linking
fn addMLIRSupport(b: *std.Build, target: *std.Build.Step.Compile, mlir_config: MLIRConfig) void {
    if (!mlir_config.enabled or mlir_config.llvm_config_path == null) {
        std.debug.print("==> Skipping MLIR support for '{s}': MLIR not enabled/found.\n", .{target.name});
        return;
    }

    std.debug.print("==> Configuring full MLIR support for '{s}'\n", .{target.name});

    // Add include and library paths from mlir_config
    if (mlir_config.include_dir) |inc_dir| {
        target.addIncludePath(.{ .cwd_relative = inc_dir });
    }
    if (mlir_config.lib_dir) |lib_dir| {
        target.addLibraryPath(.{ .cwd_relative = lib_dir });
    }

    // --- Master List of All MLIR/LLVM Libraries ---
    // This list is the consolidation of all libraries from the old addMLIRSupport,
    // m3_pipeline_test, and pass_registration_test.
    const all_mlir_libs = [_][]const u8{
        // Core MLIR & C-API
        "MLIRIR", "MLIRSupport", "MLIRAnalysis", "MLIRDialect", "MLIRParser", "MLIRAsmParser", "MLIRPass", "MLIRTransforms", "MLIRRewrite", "MLIRTransformUtils",
        "MLIRBytecodeReader", "MLIRBytecodeWriter", "MLIRCAPIIR", "MLIRCAPIFunc", "MLIRCAPIArith", "MLIRCAPILinalg", "MLIRCAPIGPU", "MLIRCAPISPIRV", "MLIRCAPISCF", "MLIRCAPIConversion", "MLIRCAPITransforms", "MLIRCAPIAsync", "MLIRCAPITensor", "MLIRCAPITransformDialect", "MLIRCAPITransformDialectTransforms",
        // Test/Debug libraries for MLIR
        // "MLIRLinalgTestPasses",

        // StableHLO & CHLO (C-API and Implementation)
        "StablehloCAPI", "ChloCAPI", "StablehloOps", "ChloOps", "StablehloBase", "StablehloPasses", "StablehloTypeInference", "StablehloAssemblyFormat", "StablehloBroadcastUtils", "StablehloLinalgTransforms", "StablehloPassUtils", "StablehloOptimizationPasses", "StablehloTypeConversion",

        // VHLO
        "VhloOps", "VhloTypes", "Version",

        // Core Dialects for Pipelines
        "MLIRFuncDialect", "MLIRArithDialect", "MLIRMathDialect", "MLIRMemRefDialect", "MLIRLinalgDialect", "MLIRTensorDialect", "MLIRSCFDialect", "MLIRVectorDialect", "MLIRAffineDialect", "MLIRBufferizationDialect", "MLIRControlFlowDialect", "MLIRAsyncDialect", "MLIRIndexDialect", "MLIRComplexDialect", "MLIRQuantDialect", "MLIRShapeDialect", "MLIRDLTIDialect", "MLIRSparseTensorDialect", "MLIRUBDialect",

        // Analysis Libraries (Critical for constraint solving)
        "MLIRPresburger", "MLIRAffineAnalysis", "MLIRAffineUtils", "MLIRDialectUtils", "MLIRArithUtils", "MLIRLinalgUtils", "MLIRSCFUtils", "MLIRTensorUtils", "MLIRTensorTilingInterfaceImpl", "MLIRMemRefUtils", "MLIRVectorUtils", "MLIRInferIntRangeCommon",

        // GPU & SPIR-V Pipeline Dialects
        "MLIRGPUDialect", "MLIRSPIRVDialect", "MLIRSPIRVSerialization", "MLIRSPIRVTarget", "MLIRSPIRVImageInterfaces", "MLIRSPIRVTransforms", "MLIRSPIRVUtils", "MLIRLLVMDialect",
        // "MLIRMeshDialect",

        // Dialect Extensions (Critical for BufferizableOpInterface)
        "MLIRFuncAllExtensions", "MLIRTensorAllExtensions", "MLIRFuncInlinerExtension",
        // "MLIRFuncMeshShardingExtensions",

        // Core Transforms & Conversion
        "MLIRFuncTransforms", "MLIRLinalgTransforms", "MLIRSCFTransforms", "MLIRGPUTransforms", "MLIRSPIRVConversion", "MLIRBufferizationTransforms", "MLIRBufferizationPipelines", "MLIRMemRefTransforms", "MLIRVectorTransforms", "MLIRArithTransforms", "MLIRAsyncTransforms", "MLIRAffineTransforms", "MLIRAsyncToLLVM", "MLIRTensorTransforms", "MLIRTensorTransformOps", "MLIRReconcileUnrealizedCasts",
        "MLIRSCFToSPIRV", "MLIRSCFToGPU", "MLIRSCFToControlFlow", "MLIRFuncToSPIRV", "MLIRMemRefToSPIRV", "MLIRVectorToSPIRV", "MLIRArithToSPIRV", "MLIRIndexToSPIRV",
        "MLIRLinalgToStandard", "MLIRConvertToLLVMPass", "MLIRConvertToLLVMInterface", "MLIRFuncToLLVM", "MLIRGPUToLLVMSPV", "MLIRLLVMCommonConversion", "MLIRArithToLLVM", "MLIRComplexToLLVM", "MLIRControlFlowToLLVM", "MLIRIndexToLLVM", "MLIRMathToLLVM", "MLIRMemRefToLLVM", "MLIRUBToLLVM", "MLIRVectorToLLVM", "MLIRAffineToStandard", "MLIRIndexingMapOpInterface",

        // GPU Pipeline Passes & Utilities  
        "MLIRGPUPipelines", "MLIRGPUToGPURuntimeTransforms", "MLIRGPUToSPIRV", "MLIRGPUUtils", "MLIRGPUToNVVMTransforms", "MLIRGPUToROCDLTransforms",
        
        // GPU Target Dialects & Translation
        "MLIRNVVMDialect", "MLIRROCDLDialect", "MLIRAMDGPUDialect", "MLIRAMDGPUUtils", "MLIRTargetLLVMIRExport", "MLIRTargetLLVMIRImport",

        // Transform Dialect for Production Tiling
        "MLIRTransformDialect", "MLIRTransformDialectTransforms", "MLIRTransformDialectUtils", "MLIRTransformDialectInterfaces", "MLIRTransformUtils", "MLIRLinalgTransformOps", "MLIRGPUTransformOps",

        // Interfaces (for vtables and dynamic dispatch) - Only actual existing libraries
        "MLIRSideEffectInterfaces", "MLIRLoopLikeInterface", "MLIRControlFlowInterfaces", "MLIRFunctionInterfaces", "MLIRShapedOpInterfaces", "MLIRViewLikeInterface", "MLIRTilingInterface", "MLIRParallelCombiningOpInterface", "MLIRDestinationStyleOpInterface", "MLIRCallInterfaces", "MLIRShardingInterface", "MLIRInferTypeOpInterface", "MLIRDataLayoutInterfaces", "MLIRCastInterfaces", "MLIRValueBoundsOpInterface", "MLIRMemorySlotInterfaces", "MLIRVectorInterfaces", "MLIRMaskableOpInterface", "MLIRMaskingOpInterface", "MLIRRuntimeVerifiableOpInterface", "MLIRSubsetOpInterface", "MLIRBytecodeOpInterface", "MLIRDerivedAttributeOpInterface", "MLIRCopyOpInterface", "MLIRInferIntRangeInterface",

        // PDL Infrastructure
        "MLIRPDLDialect", "MLIRPDLInterpDialect", "MLIRPDLToPDLInterp", "MLIRRewritePDL", "MLIRPDLLAST", "MLIRPDLLCodeGen", "MLIRTransformPDLExtension",

        // LLVM & System Libraries (Comprehensive list from your original build file)
        "LLVMWindowsManifest", "LLVMXRay", "LLVMLibDriver", "LLVMDlltoolDriver", "LLVMTelemetry", "LLVMTextAPIBinaryReader", "LLVMCoverage", "LLVMLineEditor", "LLVMAArch64Disassembler", "LLVMAArch64AsmParser", "LLVMAArch64CodeGen", "LLVMAArch64Desc", "LLVMAArch64Utils", "LLVMAArch64Info", "LLVMX86TargetMCA", "LLVMX86Disassembler", "LLVMX86AsmParser", "LLVMX86CodeGen", "LLVMX86Desc", "LLVMX86Info", "LLVMOrcDebugging", "LLVMOrcJIT", "LLVMWindowsDriver", "LLVMMCJIT", "LLVMJITLink", "LLVMInterpreter", "LLVMExecutionEngine", "LLVMRuntimeDyld", "LLVMOrcTargetProcess", "LLVMOrcShared", "LLVMDWP", "LLVMDebugInfoLogicalView", "LLVMOption", "LLVMObjCopy", "LLVMMCA", "LLVMMCDisassembler", "LLVMLTO", "LLVMPasses", "LLVMHipStdPar", "LLVMCFGuard", "LLVMCoroutines", "LLVMipo", "LLVMVectorize", "LLVMSandboxIR", "LLVMLinker", "LLVMFrontendOpenMP", "LLVMFrontendOffloading", "LLVMObjectYAML", "LLVMFrontendOpenACC", "LLVMFrontendHLSL", "LLVMFrontendDriver", "LLVMInstrumentation", "LLVMFrontendDirective", "LLVMFrontendAtomic", "LLVMExtensions", "LLVMDWARFLinkerParallel", "LLVMDWARFLinkerClassic", "LLVMDWARFLinker", "LLVMGlobalISel", "LLVMMIRParser", "LLVMAsmPrinter", "LLVMSelectionDAG", "LLVMCodeGen", "LLVMTarget", "LLVMObjCARCOpts", "LLVMCodeGenTypes", "LLVMCGData", "LLVMIRPrinter", "LLVMInterfaceStub", "LLVMFileCheck", "LLVMFuzzMutate", "LLVMScalarOpts", "LLVMInstCombine", "LLVMAggressiveInstCombine", "LLVMTransformUtils", "LLVMBitWriter", "LLVMAnalysis", "LLVMProfileData", "LLVMSymbolize", "LLVMDebugInfoBTF", "LLVMDebugInfoPDB", "LLVMDebugInfoMSF", "LLVMDebugInfoCodeView", "LLVMDebugInfoGSYM", "LLVMDebugInfoDWARF", "LLVMObject", "LLVMTextAPI", "LLVMMCParser", "LLVMIRReader", "LLVMAsmParser", "LLVMMC", "LLVMBitReader", "LLVMFuzzerCLI", "LLVMCore", "LLVMRemarks", "LLVMBitstreamReader", "LLVMBinaryFormat", "LLVMTargetParser", "LLVMTableGen", "LLVMSupport", "LLVMDemangle",
    };

    // Link C++ standard library first
    target.linkLibCpp();

    // Link all required MLIR/LLVM libraries
    for (all_mlir_libs) |lib| {
        target.linkSystemLibrary(lib);
    }

    // Platform-specific linking for Metal
    if (target.root_module.resolved_target.?.result.os.tag == .macos) {
        // Get SDK root from the environment variable set by the Nix shell.
        const sdk_root = std.process.getEnvVarOwned(b.allocator, "SDKROOT") catch |err| {
            std.debug.print("FATAL: SDKROOT not set. This requires the Nix env. Error: {}\n", .{err});
            @panic("SDKROOT not found");
        };
        defer b.allocator.free(sdk_root);

        // Construct the path to the frameworks directory inside the SDK.
        const framework_path = std.fmt.allocPrint(b.allocator, "{s}/System/Library/Frameworks", .{sdk_root}) catch @panic("OOM");
        defer b.allocator.free(framework_path);

        // **FIX:** Explicitly tell the executable's linker where to find frameworks.
        target.addFrameworkPath(.{ .cwd_relative = framework_path });

        // Link the required frameworks.
        target.linkFramework("Foundation");
        target.linkFramework("Metal");
    }
}


// Helper function to add IREE support to an executable
fn addIreeSupport(b: *std.Build, target: *std.Build.Step.Compile, iree_sdk_dir: ?[]const u8) void {
    if (iree_sdk_dir) |dir| {
        // 1. Add the include path for `iree/runtime/api.h`
        const include_path = std.fs.path.join(b.allocator, &.{dir, "include"}) catch {
            std.debug.print("Failed to construct IREE include path\n", .{});
            return;
        };
        target.addIncludePath(.{ .cwd_relative = include_path });
        
        // 2. Add the library path
        const lib_path = std.fs.path.join(b.allocator, &.{dir, "lib"}) catch {
            std.debug.print("Failed to construct IREE lib path\n", .{});
            return;
        };
        target.addLibraryPath(.{ .cwd_relative = lib_path });
        
        // 3. Link the core IREE runtime library
        target.linkSystemLibrary("iree_runtime_all_sync");
        
        std.debug.print("Added IREE support: include={s}, lib={s}\n", .{include_path, lib_path});
    } else {
        std.debug.print("IREE SDK not found, skipping IREE support\n", .{});
    }
}

pub fn build(b: *std.Build) void {
    std.debug.print("==> Starting build script\n", .{});
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    
    // Detect MLIR availability
    std.debug.print("==> Detecting MLIR configuration\n", .{});
    const mlir_config = detectMLIR(b);
    std.debug.print("==> MLIR detected: enabled={}, path={s}\n", .{mlir_config.enabled, mlir_config.llvm_config_path orelse "null"});

    // Detect Cap'n Proto availability
    std.debug.print("==> Detecting Cap'n Proto configuration\n", .{});
    const capnp_config = detectCapnp(b);
    std.debug.print("==> Cap'n Proto detected: enabled={}, include={s}, lib={s}\n", .{
        capnp_config.enabled, 
        capnp_config.include_dir orelse "null", 
        capnp_config.lib_dir orelse "null"
    });

    // Detect IREE SDK availability
    std.debug.print("==> Detecting IREE SDK configuration\n", .{});
    const iree_sdk_dir = detectIree(b);
    std.debug.print("==> IREE SDK detected: path={s}\n", .{iree_sdk_dir orelse "null"});

    // Metal bridge library removed - workers now use IREE runtime directly
    std.debug.print("==> Metal bridge library removed, using IREE runtime\n", .{});

    // SPIRV bridge library removed - workers now use IREE runtime directly
    std.debug.print("==> SPIRV bridge library removed, using IREE runtime\n", .{});


    // Create the main PCP module
    std.debug.print("==> Creating PCP module\n", .{});
    const pcp_module = b.addModule("pcp", .{
        .root_source_file = b.path("src/main.zig"),
    });
    std.debug.print("==> PCP module created successfully\n", .{});

    // Create the GPT-2 module
    const gpt2_module = b.addModule("gpt2", .{
        .root_source_file = b.path("src/models/gpt2.zig"),
    });

    // Add dependency from GPT-2 to PCP
    gpt2_module.addImport("pcp", pcp_module);

    const test_step = b.step("test", "Run unit tests");
    for (test_targets) |t_target| {
        const modules_with_tests = .{
            "src/tensor.zig",
            "src/autodiff.zig",
            "src/ops.zig",
            "src/network/message.zig",
        };
        // Add unit tests
        inline for (modules_with_tests) |module| {
            const unit_tests = b.addTest(.{
                .root_source_file = b.path(module),
                .target = b.resolveTargetQuery(t_target),
                .optimize = optimize,
            });
            const run_unit_tests = b.addRunArtifact(unit_tests);
            run_unit_tests.skip_foreign_checks = true;
            test_step.dependOn(&run_unit_tests.step);
        }
    }

    // Dialect wrapper test executable
    const dialect_test = b.addExecutable(.{
        .name = "test_dialect_wrappers",
        .root_source_file = b.path("test_dialect_wrappers.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, dialect_test, mlir_config);

    
    const run_dialect_test = b.addRunArtifact(dialect_test);
    const dialect_test_step = b.step("test-dialects", "Test MLIR dialect wrappers");
    dialect_test_step.dependOn(&run_dialect_test.step);

    // GPT-2 training example executable
    std.debug.print("==> Creating GPT-2 example executable\n", .{});
    const gpt2_example = b.addExecutable(.{
        .name = "gpt2_example",
        .root_source_file = b.path("src/examples/gpt2_training.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for GPT-2 example
    gpt2_example.root_module.addImport("pcp", pcp_module);
    gpt2_example.root_module.addImport("gpt2", gpt2_module);
    
    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, gpt2_example, mlir_config);


    // Install the executables
    //b.installArtifact(gpt2_example);

    // Run step for GPT-2 example (default)
    const run_gpt2_cmd = b.addRunArtifact(gpt2_example);
    run_gpt2_cmd.step.dependOn(&gpt2_example.step); // Only depend on the gpt2_example

    const run_gpt2_step = b.step("run", "Run the GPT-2 training example");
    run_gpt2_step.dependOn(&run_gpt2_cmd.step);


    // MLIR Comprehensive Verification Test - consolidates all MLIR functionality testing
    const mlir_verification_test = b.addExecutable(.{
        .name = "mlir_verification_test",
        .root_source_file = b.path("src/examples/mlir_verification_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies
    mlir_verification_test.root_module.addImport("pcp", pcp_module);
    
    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, mlir_verification_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries

    // Install the test executable
    //b.installArtifact(mlir_verification_test);

    // Run step for comprehensive MLIR verification
    const run_mlir_verification_cmd = b.addRunArtifact(mlir_verification_test);
    run_mlir_verification_cmd.step.dependOn(&mlir_verification_test.step);

    const run_mlir_verification_step = b.step("run-mlir-verification", "Run comprehensive MLIR operations and autodiff verification tests");
    run_mlir_verification_step.dependOn(&run_mlir_verification_cmd.step);

    // Comptime Plan examples executable
    const comptime_examples = b.addExecutable(.{
        .name = "comptime_examples",
        .root_source_file = b.path("src/examples/comptime_examples.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for comptime examples
    comptime_examples.root_module.addImport("pcp", pcp_module);
    
    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, comptime_examples, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    

    // Install the executable
    // Install disabled - failing example
    // b.installArtifact(comptime_examples);

    // Run step for comptime examples
    const run_comptime_examples_cmd = b.addRunArtifact(comptime_examples);
    run_comptime_examples_cmd.step.dependOn(b.getInstallStep());

    const run_comptime_examples_step = b.step("run-comptime-examples", "Run the comptime plan examples");
    run_comptime_examples_step.dependOn(&run_comptime_examples_cmd.step);

    // Metal backend test executable
    const metal_test = b.addExecutable(.{
        .name = "metal_test",
        .root_source_file = b.path("src/examples/metal_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for Metal test
    metal_test.root_module.addImport("pcp", pcp_module);
    
    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, metal_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    


    // Install the executable
    // Install disabled - failing example
    // b.installArtifact(metal_test);

    // Run step for Metal test
    const run_metal_test_cmd = b.addRunArtifact(metal_test);
    run_metal_test_cmd.step.dependOn(&metal_test.step);

    const run_metal_test_step = b.step("run-metal-test", "Run the Metal backend tests");
    run_metal_test_step.dependOn(&run_metal_test_cmd.step);

    // Pass registration test executable - minimal test to isolate GPU pass linking issues
    const pass_test = b.addExecutable(.{
        .name = "pass_registration_test",
        .root_source_file = b.path("src/examples/pass_registration_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies
    pass_test.root_module.addImport("pcp", pcp_module);

    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, pass_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries

    const run_pass_test_cmd = b.addRunArtifact(pass_test);
    run_pass_test_cmd.step.dependOn(&pass_test.step);

    const run_pass_test_step = b.step("run-pass-test", "Run the minimal pass registration test");
    run_pass_test_step.dependOn(&run_pass_test_cmd.step);

    // Metal benchmark executable
    const metal_benchmark = b.addExecutable(.{
        .name = "metal_benchmark",
        .root_source_file = b.path("src/examples/metal_benchmark.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for Metal benchmark
    metal_benchmark.root_module.addImport("pcp", pcp_module);
    
    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, metal_benchmark, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    


    // Install the executable
    // Install disabled - failing example  
    // b.installArtifact(metal_benchmark);

    // Run step for Metal benchmark
    const run_metal_benchmark_cmd = b.addRunArtifact(metal_benchmark);
    run_metal_benchmark_cmd.step.dependOn(&metal_benchmark.step);

    const run_metal_benchmark_step = b.step("run-metal-benchmark", "Run the Metal backend benchmarks");
    run_metal_benchmark_step.dependOn(&run_metal_benchmark_cmd.step);

    // M3 Pipeline Test - tests complete MLIR → SPIR-V → MSL → Metal pipeline
    const m3_pipeline_test = b.addExecutable(.{
        .name = "m3_pipeline_test",
        .root_source_file = b.path("src/examples/m3_pipeline_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    m3_pipeline_test.root_module.addImport("pcp", pcp_module);

    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, m3_pipeline_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries

    const run_m3_pipeline_test_cmd = b.addRunArtifact(m3_pipeline_test);
    run_m3_pipeline_test_cmd.step.dependOn(&m3_pipeline_test.step);

    const run_m3_pipeline_test_step = b.step("run-m3-pipeline-test", "Test complete MLIR → SPIR-V → MSL → Metal pipeline on M3");
    run_m3_pipeline_test_step.dependOn(&run_m3_pipeline_test_cmd.step);


    // MLIR integration test executable
    const mlir_test = b.addExecutable(.{
        .name = "mlir_test",
        .root_source_file = b.path("src/examples/mlir_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for MLIR test
    mlir_test.root_module.addImport("pcp", pcp_module);
    
    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, mlir_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    

    // Install the executable
    //b.installArtifact(mlir_test);

    // Run step for MLIR test
    const run_mlir_test_cmd = b.addRunArtifact(mlir_test);
    run_mlir_test_cmd.step.dependOn(&mlir_test.step);

    const run_mlir_test_step = b.step("run-mlir-test", "Run the MLIR integration test");
    run_mlir_test_step.dependOn(&run_mlir_test_cmd.step);

    // MLIR Tensor architecture test executable
    const tensor_mlir_test = b.addExecutable(.{
        .name = "tensor_mlir_test",
        .root_source_file = b.path("src/examples/tensor_mlir_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for tensor MLIR test
    tensor_mlir_test.root_module.addImport("pcp", pcp_module);
    
    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, tensor_mlir_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    

    // Install the executable
    // Install disabled - failing example
    // b.installArtifact(tensor_mlir_test);

    // Run step for tensor MLIR test
    const run_tensor_mlir_test_cmd = b.addRunArtifact(tensor_mlir_test);
    run_tensor_mlir_test_cmd.step.dependOn(&tensor_mlir_test.step);

    const run_tensor_mlir_test_step = b.step("run-tensor-mlir-test", "Run the MLIR tensor architecture test");
    run_tensor_mlir_test_step.dependOn(&run_tensor_mlir_test_cmd.step);

    // SPIR-V test executable
    const spirv_test = b.addExecutable(.{
        .name = "spirv_test",
        .root_source_file = b.path("test_spirv.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for SPIR-V test
    spirv_test.root_module.addImport("pcp", pcp_module);
    
    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, spirv_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries


    // Install the test executable
    //b.installArtifact(spirv_test);

    // Run step for SPIR-V test
    const run_spirv_test_cmd = b.addRunArtifact(spirv_test);
    run_spirv_test_cmd.step.dependOn(&spirv_test.step);

    const run_spirv_test_step = b.step("run-spirv-test", "Run the real SPIR-V binary generation test");
    run_spirv_test_step.dependOn(&run_spirv_test_cmd.step);

    // Distributed training system executable (main_distributed.zig)
    std.debug.print("==> Creating distributed training executable\n", .{});
    const main_distributed = b.addExecutable(.{
        .name = "main_distributed",
        .root_source_file = b.path("src/main_distributed.zig"),
        .target = target,
        .optimize = optimize,
    });
    main_distributed.root_module.addImport("pcp", pcp_module);

    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, main_distributed, mlir_config);
    
    // Add IREE support for runtime execution
    addIreeSupport(b, main_distributed, iree_sdk_dir);

    // SINGLE CALL for your project's bridge libraries

    // Cap'n Proto bridge linking (specific to this target)
    if (!capnp_config.enabled) {
        std.debug.print("==> Cap'n Proto not found, skipping distributed training system\n", .{});
    } else {
        std.debug.print("==> Adding Cap'n Proto bridge library\n", .{});
        
        const capnp_bridge_lib = b.addStaticLibrary(.{
            .name = "capnp_bridge",
            .target = target,
            .optimize = optimize,
        });
        capnp_bridge_lib.addCSourceFiles(.{
            .files = &.{
                "src/network/protocol.capnp.c++",
                "src/network/capnp_bridge.cpp",
            },
            .flags = &.{"-std=c++17"},
        });
        capnp_bridge_lib.linkLibCpp(); // IMPORTANT: Link against C++ standard library

        // NEW: Expose the public header directory to any executable that links this library.
        capnp_bridge_lib.addIncludePath(b.path("src/network"));

        // This part is for the library's own dependencies (it needs to find <capnp/c++.h>)
        if (capnp_config.include_dir) |include_dir| {
            capnp_bridge_lib.addIncludePath(.{ .cwd_relative = include_dir });
        }
        if (capnp_config.lib_dir) |lib_dir| {
            capnp_bridge_lib.addLibraryPath(.{ .cwd_relative = lib_dir });
        }
        capnp_bridge_lib.linkSystemLibrary("capnp");
        capnp_bridge_lib.linkSystemLibrary("kj");
        
        // Add include paths for the main executable
        main_distributed.addIncludePath(b.path("src/network"));
        if (capnp_config.include_dir) |include_dir| {
            main_distributed.addIncludePath(.{ .cwd_relative = include_dir });
        }

        // 2. Link the Zig executable against our bridge and the Cap'n Proto libs
        main_distributed.linkLibrary(capnp_bridge_lib);
        // Ensure the bridge library is built before the main executable
        main_distributed.step.dependOn(&capnp_bridge_lib.step);
        // Use detected Cap'n Proto libraries
        if (capnp_config.lib_dir) |lib_dir| {
            main_distributed.addLibraryPath(.{ .cwd_relative = lib_dir });
        }
        main_distributed.linkSystemLibrary("capnp");
        main_distributed.linkSystemLibrary("kj");
        
        std.debug.print("==> Cap'n Proto bridge library configured successfully\n", .{});
    }

    // Install the distributed training executable
    //b.installArtifact(main_distributed);

    // Run step for distributed training
    const run_main_distributed_cmd = b.addRunArtifact(main_distributed);
    if (b.args) |args| {
        run_main_distributed_cmd.addArgs(args);
    }
    run_main_distributed_cmd.step.dependOn(&main_distributed.step);

    std.debug.print("==> Registering run-distributed step\n", .{});
    const run_main_distributed_step = b.step("run-distributed", "Run the distributed training system");
    run_main_distributed_step.dependOn(&run_main_distributed_cmd.step);
    std.debug.print("==> run-distributed step registered successfully\n", .{});

    // Demo-only executable (no MLIR/StableHLO dependencies)
    const demo_exe = b.addExecutable(.{
        .name = "pcp_demo",
        .root_source_file = b.path("src/main_distributed_demo.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    b.installArtifact(demo_exe);
    
    const run_demo_cmd = b.addRunArtifact(demo_exe);
    if (b.args) |args| {
        run_demo_cmd.addArgs(args);
    }
    run_demo_cmd.step.dependOn(&demo_exe.step);
    
    const run_demo_step = b.step("run-demo", "Run the demo distributed training system (no MLIR)");
    run_demo_step.dependOn(&run_demo_cmd.step);

    // Data pipeline test executable
    const data_pipeline_test = b.addExecutable(.{
        .name = "data_pipeline_test",
        .root_source_file = b.path("src/examples/data_pipeline_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for data pipeline test
    data_pipeline_test.root_module.addImport("pcp", pcp_module);

    // Install the executable
    b.installArtifact(data_pipeline_test);

    // Run step for data pipeline test
    const run_data_pipeline_test_cmd = b.addRunArtifact(data_pipeline_test);
    run_data_pipeline_test_cmd.step.dependOn(&data_pipeline_test.step);

    const run_data_pipeline_test_step = b.step("test-data-pipeline", "Run the data pipeline tests");
    run_data_pipeline_test_step.dependOn(&run_data_pipeline_test_cmd.step);

    // GPT-2 model graph construction test executable
    const gpt2_model_test = b.addExecutable(.{
        .name = "gpt2_model_test",
        .root_source_file = b.path("src/examples/gpt2_model_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for GPT-2 model test
    gpt2_model_test.root_module.addImport("pcp", pcp_module);
    
    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, gpt2_model_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries

    // Install the executable
    //b.installArtifact(gpt2_model_test);

    // Run step for GPT-2 model test
    const run_gpt2_model_test_cmd = b.addRunArtifact(gpt2_model_test);
    run_gpt2_model_test_cmd.step.dependOn(&gpt2_model_test.step);

    const run_gpt2_model_test_step = b.step("run-gpt2-model-test", "Run the GPT-2 model graph construction test");
    run_gpt2_model_test_step.dependOn(&run_gpt2_model_test_cmd.step);

    // Isolated VJP Tests - Numerical verification of core autodiff rules
    const isolated_vjp_tests = b.addExecutable(.{
        .name = "isolated_vjp_tests",
        .root_source_file = b.path("src/examples/isolated_vjp_tests.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies
    isolated_vjp_tests.root_module.addImport("pcp", pcp_module);
    
    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, isolated_vjp_tests, mlir_config);

    // SINGLE CALL for your project's bridge libraries

    // Install the test executable
    //b.installArtifact(isolated_vjp_tests);

    // Run step for isolated VJP tests
    const run_isolated_vjp_tests_cmd = b.addRunArtifact(isolated_vjp_tests);
    run_isolated_vjp_tests_cmd.step.dependOn(&isolated_vjp_tests.step);

    const run_isolated_vjp_tests_step = b.step("run-isolated-vjp-tests", "Run isolated VJP numerical verification tests");
    run_isolated_vjp_tests_step.dependOn(&run_isolated_vjp_tests_cmd.step);

    // End-to-End Transformer Block Autodiff Test
    const end_to_end_transformer_test = b.addExecutable(.{
        .name = "end_to_end_transformer_test",
        .root_source_file = b.path("src/examples/end_to_end_transformer_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for end-to-end transformer test
    end_to_end_transformer_test.root_module.addImport("pcp", pcp_module);
    
    // SINGLE CALL for all MLIR/LLVM/Metal libraries
    addMLIRSupport(b, end_to_end_transformer_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries

    // Install the test executable
    //b.installArtifact(end_to_end_transformer_test);

    // Run step for end-to-end transformer test
    const run_end_to_end_transformer_test_cmd = b.addRunArtifact(end_to_end_transformer_test);
    run_end_to_end_transformer_test_cmd.step.dependOn(&end_to_end_transformer_test.step);

    const run_end_to_end_transformer_test_step = b.step("run-end-to-end-transformer-test", "Run end-to-end transformer block autodiff test with finite difference verification");
    run_end_to_end_transformer_test_step.dependOn(&run_end_to_end_transformer_test_cmd.step);

    // // Property-based testing step
    // const prop_tests = b.addTest(.{
    //     .root_source_file = b.path("src/prop_tests.zig"),
    //     .target = target,
    //     .optimize = optimize,
    // });
    //
    // const run_prop_tests = b.addRunArtifact(prop_tests);
    //
    // const run_prop_tests_step = b.step("run-prop-tests", "Run property-based tests for tensor operations");
    // run_prop_tests_step.dependOn(&run_prop_tests.step);
}
