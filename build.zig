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
fn addMLIRSupport(target: *std.Build.Step.Compile, mlir_config: MLIRConfig) void {
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

    // Add SPIRV-Cross library path if it exists
    if (std.fs.cwd().access("SPIRV-Cross/build", .{})) |_| {
        target.addLibraryPath(.{ .cwd_relative = "SPIRV-Cross/build" });
    } else |_| {
        // SPIRV-Cross not built, skip adding library path
    }

    // --- Master List of All MLIR/LLVM Libraries ---
    // This list is the consolidation of all libraries from the old addMLIRSupport,
    // m3_pipeline_test, and pass_registration_test.
    const all_mlir_libs = [_][]const u8{
        // Core MLIR & C-API
        "MLIRIR", "MLIRSupport", "MLIRAnalysis", "MLIRDialect", "MLIRParser", "MLIRAsmParser", "MLIRPass", "MLIRTransforms", "MLIRRewrite", "MLIRTransformUtils",
        "MLIRBytecodeReader", "MLIRBytecodeWriter", "MLIRCAPIIR", "MLIRCAPIFunc", "MLIRCAPIArith", "MLIRCAPILinalg", "MLIRCAPIGPU", "MLIRCAPISPIRV", "MLIRCAPISCF", "MLIRCAPIConversion", "MLIRCAPITransforms", "MLIRCAPIAsync", "MLIRCAPITensor", "MLIRCAPITransformDialect", "MLIRCAPITransformDialectTransforms",
        // Test/Debug libraries for MLIR
        "MLIRLinalgTestPasses",

        // StableHLO & CHLO (C-API and Implementation)
        "StablehloCAPI", "ChloCAPI", "StablehloOps", "ChloOps", "StablehloBase", "StablehloPasses", "StablehloTypeInference", "StablehloAssemblyFormat", "StablehloBroadcastUtils", "StablehloLinalgTransforms", "StablehloPassUtils", "StablehloOptimizationPasses", "StablehloTypeConversion",

        // VHLO
        "VhloOps", "VhloTypes", "Version",

        // Core Dialects for Pipelines
        "MLIRFuncDialect", "MLIRArithDialect", "MLIRMathDialect", "MLIRMemRefDialect", "MLIRLinalgDialect", "MLIRTensorDialect", "MLIRSCFDialect", "MLIRVectorDialect", "MLIRAffineDialect", "MLIRBufferizationDialect", "MLIRControlFlowDialect", "MLIRAsyncDialect", "MLIRIndexDialect", "MLIRComplexDialect", "MLIRQuantDialect", "MLIRShapeDialect", "MLIRDLTIDialect", "MLIRSparseTensorDialect", "MLIRUBDialect",

        // Analysis Libraries (Critical for constraint solving)
        "MLIRPresburger", "MLIRAffineAnalysis", "MLIRAffineUtils", "MLIRDialectUtils", "MLIRArithUtils", "MLIRLinalgUtils", "MLIRSCFUtils", "MLIRTensorUtils", "MLIRTensorTilingInterfaceImpl", "MLIRMemRefUtils", "MLIRVectorUtils", "MLIRInferIntRangeCommon",

        // GPU & SPIR-V Pipeline Dialects
        "MLIRGPUDialect", "MLIRSPIRVDialect", "MLIRSPIRVSerialization", "MLIRSPIRVTarget", "MLIRSPIRVImageInterfaces", "MLIRLLVMDialect", "MLIRMeshDialect",

        // Dialect Extensions (Critical for BufferizableOpInterface)
        "MLIRFuncAllExtensions", "MLIRTensorAllExtensions", "MLIRFuncInlinerExtension", "MLIRFuncMeshShardingExtensions",

        // Core Transforms & Conversion
        "MLIRFuncTransforms", "MLIRLinalgTransforms", "MLIRSCFTransforms", "MLIRGPUTransforms", "MLIRSPIRVConversion", "MLIRBufferizationTransforms", "MLIRBufferizationPipelines", "MLIRMemRefTransforms", "MLIRVectorTransforms", "MLIRArithTransforms", "MLIRAsyncTransforms", "MLIRAffineTransforms", "MLIRAsyncToLLVM", "MLIRTensorTransforms", "MLIRTensorTransformOps", "MLIRReconcileUnrealizedCasts",
        "MLIRSCFToSPIRV", "MLIRSCFToControlFlow", "MLIRFuncToSPIRV", "MLIRMemRefToSPIRV", "MLIRVectorToSPIRV", "MLIRArithToSPIRV",
        "MLIRLinalgToStandard", "MLIRConvertToLLVMPass", "MLIRFuncToLLVM", "MLIRGPUToLLVMSPV", "MLIRLLVMCommonConversion", "MLIRArithToLLVM", "MLIRComplexToLLVM", "MLIRControlFlowToLLVM", "MLIRIndexToLLVM", "MLIRMathToLLVM", "MLIRMemRefToLLVM", "MLIRUBToLLVM", "MLIRVectorToLLVM", "MLIRAffineToStandard",

        // GPU Pipeline Passes & Utilities
        "MLIRGPUPipelines", "MLIRGPUToGPURuntimeTransforms", "MLIRGPUToSPIRV", "MLIRGPUUtils",

        // Transform Dialect for Production Tiling
        "MLIRTransformDialect", "MLIRTransformDialectTransforms", "MLIRTransformDialectUtils", "MLIRTransformDialectInterfaces", "MLIRTransformUtils", "MLIRLinalgTransformOps",

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
        target.linkFramework("Foundation");
        target.linkFramework("Metal");
    }
}

// Helper function to link project's own C++/Objective-C bridge libraries
fn addPCPDependencies(target: *std.Build.Step.Compile, spirv_bridge_lib: ?*std.Build.Step.Compile, metal_bridge_lib: ?*std.Build.Step.Compile) void {
    // Link the core SPIR-V to C++ bridge library
    if (spirv_bridge_lib) |lib| {
        target.linkLibrary(lib);
    }
    // On macOS, also link the Objective-C Metal bridge library
    if (target.root_module.resolved_target.?.result.os.tag == .macos) {
        if (metal_bridge_lib) |lib| {
            target.linkLibrary(lib);
        }
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

    // Create the Metal bridge object file if on macOS
    std.debug.print("==> Creating Metal bridge library\n", .{});
    var metal_bridge_lib: ?*std.Build.Step.Compile = null;
    if (builtin.os.tag == .macos) {
        metal_bridge_lib = b.addStaticLibrary(.{
            .name = "metal_bridge",
            .target = target,
            .optimize = optimize,
        });

        metal_bridge_lib.?.addCSourceFile(.{
            .file = b.path("src/backends/metal_bridge.m"),
            .flags = &[_][]const u8{"-fobjc-arc"},
        });

        // Add frameworks for macOS
        metal_bridge_lib.?.linkFramework("Foundation");
        metal_bridge_lib.?.linkFramework("Metal");
    }

    // Create SPIR-V bridge library if MLIR is available
    std.debug.print("==> Creating SPIRV bridge library (MLIR enabled: {})\n", .{mlir_config.enabled});
    var spirv_bridge_lib: ?*std.Build.Step.Compile = null;
    if (mlir_config.enabled) {
        std.debug.print("    -> Adding SPIRV bridge static library\n", .{});
        spirv_bridge_lib = b.addStaticLibrary(.{
            .name = "spirv_bridge",
            .target = target,
            .optimize = optimize,
        });
        std.debug.print("    -> SPIRV bridge library created\n", .{});

        std.debug.print("    -> Adding SPIRV bridge source files\n", .{});
        spirv_bridge_lib.?.addCSourceFile(.{
            .file = b.path("src/mlir/spirv_bridge.cpp"),
            .flags = &[_][]const u8{"-std=c++17"},
        });
        std.debug.print("    -> Added spirv_bridge.cpp\n", .{});

        // Add SPIRV-Cross bridge for real SPIR-V → MSL translation
        std.debug.print("    -> Adding SPIRV-Cross bridge\n", .{});
        spirv_bridge_lib.?.addCSourceFile(.{
            .file = b.path("src/mlir/spirv_cross_bridge.cpp"),
            .flags = &[_][]const u8{"-std=c++17"},
        });
        std.debug.print("    -> Added spirv_cross_bridge.cpp\n", .{});

        // Add pass anchors bridge to force-load pass libraries
        std.debug.print("    -> Adding pass anchors bridge\n", .{});
        spirv_bridge_lib.?.addCSourceFile(.{
            .file = b.path("src/mlir/pass_anchors.cpp"),
            .flags = &[_][]const u8{
                "-std=c++17",
                "-Istablehlo",
                "-Illvm-build/include",
                "-Illvm-build/tools/stablehlo",
            },
        });
        std.debug.print("    -> Added pass_anchors.cpp\n", .{});

        std.debug.print("    -> Adding include paths\n", .{});
        if (mlir_config.include_dir) |include_dir| {
            spirv_bridge_lib.?.addIncludePath(.{ .cwd_relative = include_dir });
        }
        
        // Add MLIR and LLVM include directories
        spirv_bridge_lib.?.addIncludePath(.{ .cwd_relative = "llvm-project/llvm/include" }); // LLVM headers
        spirv_bridge_lib.?.addIncludePath(.{ .cwd_relative = "llvm-project/mlir/include" }); // MLIR headers
        spirv_bridge_lib.?.addIncludePath(.{ .cwd_relative = "llvm-build/tools/mlir/include" });
        spirv_bridge_lib.?.addIncludePath(.{ .cwd_relative = "llvm-build/include" });
        
        // NEW: Add StableHLO include paths
        spirv_bridge_lib.?.addIncludePath(.{ .cwd_relative = "stablehlo" });
        spirv_bridge_lib.?.addIncludePath(.{ .cwd_relative = "llvm-build/tools/stablehlo" });
        
        // Add SPIRV-Cross include paths
        spirv_bridge_lib.?.addIncludePath(.{ .cwd_relative = "SPIRV-Cross" });
        spirv_bridge_lib.?.addIncludePath(.{ .cwd_relative = "SPIRV-Cross/include" });
        // Add our local MLIR library directory FIRST  
        if (mlir_config.lib_dir) |lib_dir| {
            spirv_bridge_lib.?.addLibraryPath(.{ .cwd_relative = lib_dir });
            std.debug.print("    -> Added MLIR library path: {s}\n", .{lib_dir});
        } else {
            // Fallback to project-relative MLIR build directory
            spirv_bridge_lib.?.addLibraryPath(.{ .cwd_relative = "llvm-build/lib" });
            std.debug.print("    -> Added fallback MLIR library path: llvm-build/lib\n", .{});
        }
        
        std.debug.print("    -> Adding MLIR libraries\n", .{});
        // Core MLIR/LLVM libraries (always required)
        const core_spirv_libs = [_][]const u8{
            "MLIRCAPIIR",
            "MLIRIR",
            "MLIRSPIRVDialect",
            "MLIRSPIRVSerialization",
            "MLIRSPIRVBinaryUtils",
            "MLIRSupport",
            "LLVMSupport",
            "LLVMCore",
        };
        
        std.debug.print("    -> Linking core SPIRV libraries ({} libraries)\n", .{core_spirv_libs.len});
        for (core_spirv_libs) |lib| {
            std.debug.print("      -> Linking: {s}\n", .{lib});
            spirv_bridge_lib.?.linkSystemLibrary(lib);
        }
        std.debug.print("    -> Core SPIRV libraries linked successfully\n", .{});
        
        // Add SPIRV-Cross libraries if they're built
        const spirv_cross_build_dir = std.fs.path.join(b.allocator, &[_][]const u8{ "SPIRV-Cross", "build" }) catch {
            std.debug.print("Failed to construct SPIRV-Cross build path\n", .{});
            return;
        };
        defer b.allocator.free(spirv_cross_build_dir);
        
        std.fs.cwd().access(spirv_cross_build_dir, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                std.debug.print("Warning: SPIRV-Cross not built. Some features may be unavailable.\n", .{});
                std.debug.print("To build SPIRV-Cross: cd SPIRV-Cross && mkdir -p build && cd build && cmake .. && make\n", .{});
                return;
            },
            else => {
                std.debug.print("Warning: Cannot access SPIRV-Cross build directory: {}\n", .{err});
                return;
            },
        };
        
        // Add SPIRV-Cross library path and libraries (optional)
        spirv_bridge_lib.?.addLibraryPath(.{ .cwd_relative = "SPIRV-Cross/build" });
        
        const spirv_cross_libs = [_][]const u8{
            "spirv-cross-msl",
            "spirv-cross-glsl",        // MSL depends on GLSL
            "spirv-cross-hlsl",        // C API uses HLSL compiler
            "spirv-cross-cpp",         // C API uses CPP compiler  
            "spirv-cross-reflect",     // C API uses reflection
            "spirv-cross-core",
            "spirv-cross-c",
        };
        
        std.debug.print("    -> Linking SPIRV-Cross libraries...\n", .{});
        
        // Only link SPIRV-Cross libraries if the build directory exists
        if (std.fs.cwd().access("SPIRV-Cross/build", .{})) |_| {
            std.debug.print("    -> SPIRV-Cross build found, linking libraries\n", .{});
            for (spirv_cross_libs) |lib| {
                spirv_bridge_lib.?.linkSystemLibrary(lib);
            }
            std.debug.print("    -> Successfully linked SPIRV-Cross libraries\n", .{});
        } else |err| switch (err) {
            error.FileNotFound => {
                std.debug.print("    -> SPIRV-Cross not built, skipping library linking\n", .{});
            },
            else => {
                std.debug.print("    -> Error accessing SPIRV-Cross directory: {}\n", .{err});
            },
        }
        
        spirv_bridge_lib.?.linkLibCpp();
        std.debug.print("    -> SPIRV bridge library configuration completed\n", .{});
    }
    std.debug.print("==> SPIRV bridge section completed\n", .{});


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
            "src/optimizers/adam.zig",
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
    addMLIRSupport(dialect_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    addPCPDependencies(dialect_test, spirv_bridge_lib, metal_bridge_lib);
    
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
    addMLIRSupport(gpt2_example, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    addPCPDependencies(gpt2_example, spirv_bridge_lib, metal_bridge_lib);

    // Install the executables
    b.installArtifact(gpt2_example);

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
    addMLIRSupport(mlir_verification_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    addPCPDependencies(mlir_verification_test, spirv_bridge_lib, metal_bridge_lib);

    // Install the test executable
    b.installArtifact(mlir_verification_test);

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
    addMLIRSupport(comptime_examples, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    addPCPDependencies(comptime_examples, spirv_bridge_lib, metal_bridge_lib);
    

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
    addMLIRSupport(metal_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    addPCPDependencies(metal_test, spirv_bridge_lib, metal_bridge_lib);
    


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
    addMLIRSupport(pass_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    addPCPDependencies(pass_test, spirv_bridge_lib, metal_bridge_lib);

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
    addMLIRSupport(metal_benchmark, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    addPCPDependencies(metal_benchmark, spirv_bridge_lib, metal_bridge_lib);
    

    // Special case: metal_benchmark needs additional Metal source file
    if (builtin.os.tag == .macos) {
        // Set dynamic link flags to ensure visibility of Metal symbols
        metal_benchmark.addCSourceFile(.{
            .file = b.path("src/backends/metal_bridge.m"),
            .flags = &[_][]const u8{"-fobjc-arc"},
        });
    }

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
    addMLIRSupport(m3_pipeline_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    addPCPDependencies(m3_pipeline_test, spirv_bridge_lib, metal_bridge_lib);

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
    addMLIRSupport(mlir_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    addPCPDependencies(mlir_test, spirv_bridge_lib, metal_bridge_lib);
    

    // Install the executable
    b.installArtifact(mlir_test);

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
    addMLIRSupport(tensor_mlir_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    addPCPDependencies(tensor_mlir_test, spirv_bridge_lib, metal_bridge_lib);
    

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
    addMLIRSupport(spirv_test, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    addPCPDependencies(spirv_test, spirv_bridge_lib, metal_bridge_lib);


    // Install the test executable
    b.installArtifact(spirv_test);

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
    addMLIRSupport(main_distributed, mlir_config);

    // SINGLE CALL for your project's bridge libraries
    addPCPDependencies(main_distributed, spirv_bridge_lib, metal_bridge_lib);

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
    b.installArtifact(main_distributed);

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
