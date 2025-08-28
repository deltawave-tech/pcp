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

// MLIR support using manual library linking
fn addMLIRSupport(b: *std.Build, target: *std.Build.Step.Compile, mlir_config: MLIRConfig) void {
    if (!mlir_config.enabled) {
        std.debug.print("==> Skipping MLIR support for '{s}': MLIR not enabled/found.\n", .{target.name});
        return; // Return gracefully instead of panicking
    }
    
    const llvm_config_path = mlir_config.llvm_config_path orelse {
        std.debug.print("==> Skipping MLIR support for '{s}': llvm-config path is null.\n", .{target.name});
        return; // Return gracefully instead of panicking
    };

    std.debug.print("==> Configuring MLIR support for '{s}' using '{s}'\n", .{target.name, llvm_config_path});
    
    // FORCE use of local ARM64 libraries by prioritizing our paths
    // Add our local library path FIRST to override system libraries
    if (mlir_config.lib_dir) |lib_dir| {
        target.addLibraryPath(.{ .cwd_relative = lib_dir });
    }
    
    // Add ARM64 Homebrew path (if exists) before x86_64 paths
    target.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    
    // Try llvm-config but don't fail if it doesn't work
    std.debug.print("    -> Querying for system libs...\n", .{});
    const system_libs = getCommandOutput(b, &[_][]const u8{llvm_config_path, "--system-libs"}) catch blk: {
        std.debug.print("    -> FAILED to get system libs from llvm-config. Continuing without them.\n", .{});
        break :blk "";
    };
    std.debug.print("    -> Found system libs: {s}\n", .{system_libs});

    std.debug.print("    -> Querying for ldflags...\n", .{});
    const ldflags = getCommandOutput(b, &[_][]const u8{llvm_config_path, "--ldflags"}) catch blk: {
        std.debug.print("    -> FAILED to get ldflags from llvm-config. Continuing without them.\n", .{});
        break :blk "";
    };
    std.debug.print("    -> Found ldflags: {s}\n", .{ldflags});
    
    // Link system libraries if available (but our paths take precedence)
    if (system_libs.len > 0) {
        var sys_iter = std.mem.splitSequence(u8, system_libs, " ");
        while (sys_iter.next()) |lib_flag| {
            if (lib_flag.len > 0) {
                if (std.mem.startsWith(u8, lib_flag, "-l")) {
                    target.linkSystemLibrary(lib_flag[2..]);
                }
            }
        }
    }

    // Add linker flags if available (but skip x86_64 paths)
    if (ldflags.len > 0) {
        var flag_iter = std.mem.splitSequence(u8, ldflags, " ");
        while (flag_iter.next()) |flag| {
            if (flag.len > 0 and std.mem.startsWith(u8, flag, "-L")) {
                // Skip x86_64 paths that would cause architecture conflicts
                const path = flag[2..];
                if (!std.mem.startsWith(u8, path, "/usr/local")) {
                    target.addLibraryPath(.{ .cwd_relative = path });
                }
            }
        }
    }
    
    // Add include directory
    if (mlir_config.include_dir) |inc_dir| {
        target.addIncludePath(.{ .cwd_relative = inc_dir });
    }
    
    // Add library directory
    if (mlir_config.lib_dir) |lib_dir| {
        target.addLibraryPath(.{ .cwd_relative = lib_dir });
    }
    
    // Add SPIRV-Cross library directory if it exists
    if (std.fs.cwd().access("SPIRV-Cross/build", .{})) |_| {
        target.addLibraryPath(.{ .cwd_relative = "SPIRV-Cross/build" });
    } else |err| switch (err) {
        error.FileNotFound => {
            // SPIRV-Cross not built, skip adding library path
        },
        else => {
            // Some other error accessing the directory
        },
    }
    
    // Selective library linking - core MLIR libraries plus specific dialects and passes
    const mlir_libs = [_][]const u8{
        // Core MLIR infrastructure
        "MLIRIR",
        "MLIRSupport",
        "MLIRAnalysis",
        "MLIRDialect",
        "MLIRParser",
        "MLIRPass",
        "MLIRTransforms",
        
        // Bytecode and parsing support
        "MLIRBytecodeReader",
        "MLIRBytecodeWriter",
        "MLIRBytecodeOpInterface",
        "MLIRAsmParser",
        
        // C-API libraries for dialect registration (CRITICAL for our dialect handles)
        "MLIRCAPIIR",
        "MLIRCAPIFunc",
        "MLIRCAPIArith",
        "MLIRCAPILinalg",
        "MLIRCAPIGPU",
        "MLIRCAPISPIRV",
        "MLIRCAPISCF",
        // StableHLO C-API libraries
        "StablehloCAPI",
        "ChloCAPI",
        // StableHLO dialect implementation libraries
        "StablehloOps",
        "ChloOps",
        "StablehloTypeInference",
        "StablehloAssemblyFormat",
        "StablehloBase",
        "StablehloBroadcastUtils",
        
        // Specific dialects for our pipeline (using actual library names)
        "MLIRLinalgDialect",
        "MLIRGPUDialect",
        "MLIRSPIRVDialect",
        // SPIR-V target libraries for real serialization
        "MLIRSPIRVSerialization",
        "MLIRSPIRVTarget",
        "MLIRSPIRVBinaryUtils",
        "MLIRSCFDialect",
        "MLIRFuncDialect",
        "MLIRArithDialect",
        "MLIRBufferizationDialect",
        "MLIRTensorDialect",
        "MLIRMemRefDialect",
        "MLIRMathDialect",
        "MLIRIndexDialect",
        "MLIRComplexDialect",
        "MLIRSparseTensorDialect",
        "MLIRAffineDialect",
        "MLIRControlFlowDialect",
        "MLIRUBDialect",
        "MLIRDLTIDialect",
        "MLIRQuantDialect",
        "MLIRShapeDialect",
        
        // Additional required libraries for missing symbols
        "MLIRDialectUtils",
        "MLIRArithUtils",
        "MLIRLinalgUtils",
        "MLIRAffineUtils",        // For affine utility functions
        "MLIRAffineAnalysis",     // For affine analysis (MemRefRegion, FlatAffineValueConstraints)
        "MLIRSCFUtils",          // For SCF utility functions
        "MLIRTensorUtils",       // For tensor utility functions
        "MLIRTensorTilingInterfaceImpl", // For bubbleUpPadSlice  
        "MLIRLinalgTransforms",
        "MLIRBufferizationTransforms",
        "MLIRTensorTransforms",  // Already included - may contain bubbleUpPadSlice
        "MLIRMemRefTransforms",
        
        // Interface libraries for missing vtable symbols
        "MLIRSideEffectInterfaces",
        "MLIRControlFlowInterfaces",
        "MLIRInferIntRangeInterface",
        "MLIRViewLikeInterface",
        "MLIRDestinationStyleOpInterface",
        "MLIRFunctionInterfaces",
        "MLIRShapedOpInterfaces",
        "MLIRTilingInterface",
        "MLIRCallInterfaces",
        "MLIRValueBoundsOpInterface",
        "MLIRPresburger",
        "MLIRParallelCombiningOpInterface",
        "MLIRDataLayoutInterfaces",
        "MLIRCastInterfaces",
        "MLIRSPIRVImageInterfaces",
        
        // Additional interface libraries for missing symbols
        "MLIRInferIntRangeCommon",
        "MLIRInferTypeOpInterface",
        "MLIRLoopLikeInterface",
        "MLIRMemorySlotInterfaces",
        
        // Pass management
        "MLIRPass",
        
        // Pass registration libraries (GRANULAR - minimal dependencies)
        "StablehloPasses",            // Contains mlirRegisterStablehloLegalizeToLinalgPass
        "MLIRTransforms",             // Contains the actual pass implementations
        "MLIRCAPITransforms",         // Contains mlirRegisterTransformsCanonicalizer and mlirRegisterTransformsCSE
        
        // NEW: Additional pass libraries for complete GPU lowering pipeline (corrected names)
        "MLIRLinalgTransforms",       // For linalg-bufferize, linalg-to-parallel-loops
        "MLIRGPUTransforms",          // For gpu-map-parallel-loops, convert-gpu-to-spirv  
        "MLIRSCFTransforms",          // For SCF dialect passes
        "MLIRAffineTransforms",       // For lower-affine
        "MLIRMemRefTransforms",       // For bufferization and memref passes (already included)
        "MLIRCAPIConversion",         // For conversion pass registration
        "MLIRConvertToLLVMPass",     // For general LLVM conversion infrastructure
        "MLIRFuncToLLVM",            // For createConvertFuncToLLVMPass
        "MLIRGPUToLLVMSPV",          // For GPU to SPIRV LLVM conversion
        "MLIRGPUPipelines",          // For createGpuToLLVMConversionPass
        "MLIRGPUToGPURuntimeTransforms", // Additional GPU to LLVM transforms
        "MLIRReconcileUnrealizedCasts", // For createReconcileUnrealizedCastsPass
        
        // Dependencies of the passes above
        "MLIRLinalgDialect",          // Target dialect for the legalization
        "MLIRVectorDialect",          // Vector dialect for vector operations
        "MLIRVectorTransforms",       // Vector dialect transforms
        "MLIRVectorUtils",            // Vector dialect utilities
        "MLIRVectorInterfaces",       // Vector interfaces
        "MLIRMaskableOpInterface",    // Maskable operation interfaces
        "MLIRMaskingOpInterface",     // Masking operation interfaces
        "StablehloOps",               // Source dialect for the legalization
        "StablehloLinalgTransforms",  // For StableHLO to Linalg conversion
        "VhloOps",                    // VHLO operations needed by StableHLO passes
        "VhloTypes",                  // VHLO types
        
        // KEY: Pattern Rewrite and Conversion Infrastructure
        "MLIRRewrite",                // Core pattern rewrite logic (FrozenRewritePatternSet, etc.)
        "MLIRTransformUtils",         // Transform utilities (applyPatternsGreedily, etc.)
        
        // Common Interface and Utility Dependencies
        "MLIRSideEffectInterfaces",   // Very common dependency
        "MLIRLoopLikeInterface",      // For loops (affine, scf)
        "MLIRDialectUtils",           // Common dialect helper functions
        "MLIRAnalysis",               // Core analysis infrastructure
        
        // Additional interfaces needed by StableHLO passes
        "MLIRSubsetOpInterface",      // For subset operations
        "MLIRPDLToPDLInterp",         // For PDL pattern conversion
        "MLIRRuntimeVerifiableOpInterface", // For runtime verification
        "MLIRPDLInterpDialect",       // PDL interpreter dialect
        "MLIRPDLDialect",             // PDL dialect
        "MLIRFuncTransforms",         // Function transformation patterns
        "StablehloPassUtils",         // StableHLO utility functions
        "Version",                    // VHLO version utilities
        
        // PDL and StableHLO optimizations
        "MLIRRewritePDL",             // PDL bytecode constructor and rewrite functions
        "StablehloOptimizationPasses", // StableHLO shape folder patterns
        "StablehloTypeConversion",    // RemoveSignTypeConverter functions
        
        // Additional MLIR conversion libraries for LLVM conversion
        "MLIRLLVMDialect",           // LLVM dialect for MLIR
        "MLIRLLVMCommonConversion",  // Common LLVM conversion infrastructure 
        "MLIRLLVMToLLVMIRTranslation", // LLVM to LLVM IR translation
        "MLIRToLLVMIRTranslationRegistration", // LLVM IR translation registration  
        "MLIRFromLLVMIRTranslationRegistration", // From LLVM IR translation registration
        "MLIRBuiltinToLLVMIRTranslation", // Builtin to LLVM IR translation
        "MLIRTargetLLVMIRExport",    // LLVM IR export and TypeToLLVMIRTranslator
        "MLIRAsyncDialect",          // Async dialect for async operations
        "MLIRAsyncTransforms",       // Async dialect transforms
        "MLIRAsyncToLLVM",           // Async to LLVM conversion
        "MLIRVectorToLLVM",          // Vector to LLVM conversion
        "MLIRArithToLLVM",           // Arith to LLVM conversion
        "MLIRArithAttrToLLVMConversion", // Arith attributes to LLVM conversion
        "MLIRControlFlowToLLVM",     // Control flow to LLVM conversion
        "MLIRIndexToLLVM",           // Index to LLVM conversion
        "MLIRMemRefToLLVM",          // MemRef to LLVM conversion
        "MLIRMemRefUtils",           // MemRef utilities
        "MLIRMathToLLVM",            // Math to LLVM conversion
        "MLIRComplexToLLVM",         // Complex to LLVM conversion
        "MLIRUBToLLVM",              // UB (undefined behavior) to LLVM conversion
        "MLIROpenMPToLLVM",          // OpenMP to LLVM conversion
        "MLIRConvertToLLVMInterface", // LLVM conversion interfaces
        
        // Complete LLVM libraries from llvm-config --libs
        "LLVMWindowsManifest", "LLVMXRay", "LLVMLibDriver", "LLVMDlltoolDriver", "LLVMTelemetry", "LLVMTextAPIBinaryReader",
        "LLVMCoverage", "LLVMLineEditor", "LLVMAArch64Disassembler", "LLVMAArch64AsmParser", "LLVMAArch64CodeGen", 
        "LLVMAArch64Desc", "LLVMAArch64Utils", "LLVMAArch64Info", "LLVMX86TargetMCA", "LLVMX86Disassembler", 
        "LLVMX86AsmParser", "LLVMX86CodeGen", "LLVMX86Desc", "LLVMX86Info", "LLVMOrcDebugging", "LLVMOrcJIT", 
        "LLVMWindowsDriver", "LLVMMCJIT", "LLVMJITLink", "LLVMInterpreter", "LLVMExecutionEngine", "LLVMRuntimeDyld", 
        "LLVMOrcTargetProcess", "LLVMOrcShared", "LLVMDWP", "LLVMDebugInfoLogicalView", "LLVMOption", "LLVMObjCopy", 
        "LLVMMCA", "LLVMMCDisassembler", "LLVMLTO", "LLVMPasses", "LLVMHipStdPar", "LLVMCFGuard", "LLVMCoroutines", 
        "LLVMipo", "LLVMVectorize", "LLVMSandboxIR", "LLVMLinker", "LLVMFrontendOpenMP", "LLVMFrontendOffloading", 
        "LLVMObjectYAML", "LLVMFrontendOpenACC", "LLVMFrontendHLSL", "LLVMFrontendDriver", "LLVMInstrumentation", 
        "LLVMFrontendDirective", "LLVMFrontendAtomic", "LLVMExtensions", "LLVMDWARFLinkerParallel", "LLVMDWARFLinkerClassic", 
        "LLVMDWARFLinker", "LLVMGlobalISel", "LLVMMIRParser", "LLVMAsmPrinter", "LLVMSelectionDAG", "LLVMCodeGen", 
        "LLVMTarget", "LLVMObjCARCOpts", "LLVMCodeGenTypes", "LLVMCGData", "LLVMIRPrinter", "LLVMInterfaceStub", 
        "LLVMFileCheck", "LLVMFuzzMutate", "LLVMScalarOpts", "LLVMInstCombine", "LLVMAggressiveInstCombine", 
        "LLVMTransformUtils", "LLVMBitWriter", "LLVMAnalysis", "LLVMProfileData", "LLVMSymbolize", "LLVMDebugInfoBTF", 
        "LLVMDebugInfoPDB", "LLVMDebugInfoMSF", "LLVMDebugInfoCodeView", "LLVMDebugInfoGSYM", "LLVMDebugInfoDWARF", 
        "LLVMObject", "LLVMTextAPI", "LLVMMCParser", "LLVMIRReader", "LLVMAsmParser", "LLVMMC", "LLVMBitReader", 
        "LLVMFuzzerCLI", "LLVMCore", "LLVMRemarks", "LLVMBitstreamReader", "LLVMBinaryFormat", "LLVMTargetParser", 
        "LLVMTableGen", "LLVMSupport", "LLVMDemangle",
    };
    
    // Link C++ standard library first
    target.linkLibCpp();
    
    // Link MLIR/LLVM libraries
    for (mlir_libs) |lib| {
        target.linkSystemLibrary(lib);
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
    
    addMLIRSupport(b, dialect_test, mlir_config);
    
    if (spirv_bridge_lib != null) {
        dialect_test.linkLibrary(spirv_bridge_lib.?);
    }
    
    if (builtin.os.tag == .macos and metal_bridge_lib != null) {
        dialect_test.linkLibrary(metal_bridge_lib.?);
    }
    
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
    
    // Add MLIR support to GPT-2 example
    addMLIRSupport(b, gpt2_example, mlir_config);
    
    if (spirv_bridge_lib != null) {
        gpt2_example.linkLibrary(spirv_bridge_lib.?);
    }

    // Install the executables
    b.installArtifact(gpt2_example);

    // Run step for GPT-2 example (default)
    const run_gpt2_cmd = b.addRunArtifact(gpt2_example);
    run_gpt2_cmd.step.dependOn(&gpt2_example.step); // Only depend on the gpt2_example

    const run_gpt2_step = b.step("run", "Run the GPT-2 training example");
    run_gpt2_step.dependOn(&run_gpt2_cmd.step);

    // Autodiff test executable
    const autodiff_test = b.addExecutable(.{
        .name = "autodiff_test",
        .root_source_file = b.path("src/examples/autodiff_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for autodiff test
    autodiff_test.root_module.addImport("pcp", pcp_module);
    
    // Add MLIR support to autodiff test
    addMLIRSupport(b, autodiff_test, mlir_config);
    
    if (spirv_bridge_lib != null) {
        autodiff_test.linkLibrary(spirv_bridge_lib.?);
    }

    // Install disabled - failing example
    // b.installArtifact(autodiff_test);

    // Run step for autodiff test
    const run_autodiff_test_cmd = b.addRunArtifact(autodiff_test);
    run_autodiff_test_cmd.step.dependOn(&autodiff_test.step);

    const run_autodiff_test_step = b.step("run-autodiff-test", "Run the autodiff Plan-based tests");
    run_autodiff_test_step.dependOn(&run_autodiff_test_cmd.step);

    // MLIR Autodiff Simple Test executable
    const mlir_autodiff_simple_test = b.addExecutable(.{
        .name = "mlir_autodiff_simple_test",
        .root_source_file = b.path("src/examples/mlir_autodiff_simple_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for MLIR autodiff simple test
    mlir_autodiff_simple_test.root_module.addImport("pcp", pcp_module);
    
    // Add MLIR support to MLIR autodiff simple test
    addMLIRSupport(b, mlir_autodiff_simple_test, mlir_config);
    
    if (spirv_bridge_lib != null) {
        mlir_autodiff_simple_test.linkLibrary(spirv_bridge_lib.?);
    }
    

    // Install the test executable
    b.installArtifact(mlir_autodiff_simple_test);

    // Run step for MLIR autodiff simple test
    const run_mlir_autodiff_simple_test_cmd = b.addRunArtifact(mlir_autodiff_simple_test);
    run_mlir_autodiff_simple_test_cmd.step.dependOn(&mlir_autodiff_simple_test.step);

    const run_mlir_autodiff_simple_test_step = b.step("run-mlir-autodiff-simple-test", "Run the MLIR autodiff simple test");
    run_mlir_autodiff_simple_test_step.dependOn(&run_mlir_autodiff_simple_test_cmd.step);

    // MLIR autodiff matmul test executable
    const mlir_autodiff_matmul_test = b.addExecutable(.{
        .name = "mlir_autodiff_matmul_test",
        .root_source_file = b.path("src/examples/mlir_autodiff_matmul_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for MLIR autodiff matmul test
    mlir_autodiff_matmul_test.root_module.addImport("pcp", pcp_module);
    
    // Add MLIR support to MLIR autodiff matmul test
    addMLIRSupport(b, mlir_autodiff_matmul_test, mlir_config);
    
    if (spirv_bridge_lib != null) {
        mlir_autodiff_matmul_test.linkLibrary(spirv_bridge_lib.?);
    }
    

    // Install the test executable
    b.installArtifact(mlir_autodiff_matmul_test);

    // Run step for MLIR autodiff matmul test
    const run_mlir_autodiff_matmul_test_cmd = b.addRunArtifact(mlir_autodiff_matmul_test);
    run_mlir_autodiff_matmul_test_cmd.step.dependOn(&mlir_autodiff_matmul_test.step);

    const run_mlir_autodiff_matmul_test_step = b.step("run-mlir-autodiff-matmul-test", "Run the MLIR autodiff matmul test");
    run_mlir_autodiff_matmul_test_step.dependOn(&run_mlir_autodiff_matmul_test_cmd.step);

    // MLIR autodiff verification test executable
    const mlir_autodiff_verification_test = b.addExecutable(.{
        .name = "mlir_autodiff_verification_test",
        .root_source_file = b.path("src/examples/mlir_autodiff_verification_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for MLIR autodiff verification test
    mlir_autodiff_verification_test.root_module.addImport("pcp", pcp_module);
    
    // Add MLIR support to MLIR autodiff verification test
    addMLIRSupport(b, mlir_autodiff_verification_test, mlir_config);
    
    if (spirv_bridge_lib != null) {
        mlir_autodiff_verification_test.linkLibrary(spirv_bridge_lib.?);
    }
    

    // Link Metal bridge on macOS
    if (builtin.os.tag == .macos) {
        mlir_autodiff_verification_test.linkFramework("Foundation");
        mlir_autodiff_verification_test.linkFramework("Metal");
        mlir_autodiff_verification_test.linkLibrary(metal_bridge_lib.?);
    }

    // Install the test executable
    b.installArtifact(mlir_autodiff_verification_test);

    // Run step for MLIR autodiff verification test
    const run_mlir_autodiff_verification_test_cmd = b.addRunArtifact(mlir_autodiff_verification_test);
    run_mlir_autodiff_verification_test_cmd.step.dependOn(&mlir_autodiff_verification_test.step);

    const run_mlir_autodiff_verification_test_step = b.step("run-mlir-autodiff-verification-test", "Run the MLIR autodiff verification test");
    run_mlir_autodiff_verification_test_step.dependOn(&run_mlir_autodiff_verification_test_cmd.step);

    // Simple multiply test executable
    const mlir_simple_multiply_test = b.addExecutable(.{
        .name = "mlir_simple_multiply_test",
        .root_source_file = b.path("src/examples/mlir_simple_multiply_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for simple multiply test
    mlir_simple_multiply_test.root_module.addImport("pcp", pcp_module);
    
    // Add MLIR support to simple multiply test
    addMLIRSupport(b, mlir_simple_multiply_test, mlir_config);
    
    if (spirv_bridge_lib != null) {
        mlir_simple_multiply_test.linkLibrary(spirv_bridge_lib.?);
    }
    

    // Install the test executable
    b.installArtifact(mlir_simple_multiply_test);

    // Run step for simple multiply test
    const run_mlir_simple_multiply_test_cmd = b.addRunArtifact(mlir_simple_multiply_test);
    run_mlir_simple_multiply_test_cmd.step.dependOn(&mlir_simple_multiply_test.step);

    const run_mlir_simple_multiply_test_step = b.step("run-mlir-simple-multiply-test", "Run the MLIR simple multiply test");
    run_mlir_simple_multiply_test_step.dependOn(&run_mlir_simple_multiply_test_cmd.step);

    // MLIR autodiff numerical test executable
    const mlir_autodiff_numerical_test = b.addExecutable(.{
        .name = "mlir_autodiff_numerical_test",
        .root_source_file = b.path("src/examples/mlir_autodiff_numerical_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for numerical test
    mlir_autodiff_numerical_test.root_module.addImport("pcp", pcp_module);
    
    // Add MLIR support to numerical test
    addMLIRSupport(b, mlir_autodiff_numerical_test, mlir_config);
    
    if (spirv_bridge_lib != null) {
        mlir_autodiff_numerical_test.linkLibrary(spirv_bridge_lib.?);
    }

    // Link Metal bridge on macOS
    if (builtin.os.tag == .macos) {
        mlir_autodiff_numerical_test.linkFramework("Foundation");
        mlir_autodiff_numerical_test.linkFramework("Metal");
        mlir_autodiff_numerical_test.linkLibrary(metal_bridge_lib.?);
    }

    // Install the test executable
    b.installArtifact(mlir_autodiff_numerical_test);

    // Run step for numerical test
    const run_mlir_autodiff_numerical_test_cmd = b.addRunArtifact(mlir_autodiff_numerical_test);
    run_mlir_autodiff_numerical_test_cmd.step.dependOn(&mlir_autodiff_numerical_test.step);

    const run_mlir_autodiff_numerical_test_step = b.step("run-mlir-autodiff-numerical-test", "Run the MLIR autodiff numerical verification test");
    run_mlir_autodiff_numerical_test_step.dependOn(&run_mlir_autodiff_numerical_test_cmd.step);

    // Comptime Plan examples executable
    const comptime_examples = b.addExecutable(.{
        .name = "comptime_examples",
        .root_source_file = b.path("src/examples/comptime_examples.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for comptime examples
    comptime_examples.root_module.addImport("pcp", pcp_module);
    
    // Add MLIR support to comptime examples
    addMLIRSupport(b, comptime_examples, mlir_config);
    
    if (spirv_bridge_lib != null) {
        comptime_examples.linkLibrary(spirv_bridge_lib.?);
    }
    

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
    
    // Add MLIR support to metal test
    addMLIRSupport(b, metal_test, mlir_config);
    
    if (spirv_bridge_lib != null) {
        metal_test.linkLibrary(spirv_bridge_lib.?);
    }
    

    // Link Metal bridge on macOS
    if (builtin.os.tag == .macos) {
        metal_test.linkFramework("Foundation");
        metal_test.linkFramework("Metal");
        metal_test.linkLibrary(metal_bridge_lib.?);
    }

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

    // Add MLIR support with additional GPU libraries for testing
    addMLIRSupport(b, pass_test, mlir_config);

    // Add the suspected missing libraries for GPU passes
    if (mlir_config.enabled) {
        // Core dialects
        pass_test.linkSystemLibrary("MLIRLinalgDialect");
        pass_test.linkSystemLibrary("MLIRSCFDialect");
        pass_test.linkSystemLibrary("MLIRGPUDialect");
        
        // GPU-related passes and transforms
        pass_test.linkSystemLibrary("MLIRGPUTransforms");
        pass_test.linkSystemLibrary("MLIRGPUPipelines");
        pass_test.linkSystemLibrary("MLIRGPUToGPURuntimeTransforms");
        
        // Linalg transforms and conversions
        pass_test.linkSystemLibrary("MLIRLinalgTransforms");
        pass_test.linkSystemLibrary("MLIRLinalgToStandard");
        
        // SCF transforms  
        pass_test.linkSystemLibrary("MLIRSCFTransforms");
        
        // GPU to SPIR-V conversion
        pass_test.linkSystemLibrary("MLIRGPUToSPIRV");
        
        // Additional interfaces and utilities
        pass_test.linkSystemLibrary("MLIRParallelCombiningOpInterface");
        pass_test.linkSystemLibrary("MLIRSPIRVConversion");
        pass_test.linkSystemLibrary("MLIRCAPIConversion");
        
        // GPU utils and interfaces
        pass_test.linkSystemLibrary("MLIRGPUUtils");
    }

    if (spirv_bridge_lib != null) {
        pass_test.linkLibrary(spirv_bridge_lib.?);
    }

    if (builtin.os.tag == .macos and metal_bridge_lib != null) {
        pass_test.linkFramework("Foundation");
        pass_test.linkFramework("Metal");
        pass_test.linkLibrary(metal_bridge_lib.?);
    }

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
    
    // Add MLIR support to metal benchmark
    addMLIRSupport(b, metal_benchmark, mlir_config);
    
    if (spirv_bridge_lib != null) {
        metal_benchmark.linkLibrary(spirv_bridge_lib.?);
    }
    

    // Link Metal bridge on macOS with extra debugging
    if (builtin.os.tag == .macos) {
        std.debug.print("Adding Metal frameworks to benchmark...\n", .{});
        metal_benchmark.linkFramework("Foundation");
        metal_benchmark.linkFramework("Metal");
        metal_benchmark.linkLibrary(metal_bridge_lib.?);

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

    // Add module dependencies
    m3_pipeline_test.root_module.addImport("pcp", pcp_module);
    
    // Add MLIR support
    addMLIRSupport(b, m3_pipeline_test, mlir_config);
    
    // Add the same GPU libraries as pass_registration_test
    if (mlir_config.enabled) {
        // Core dialects
        m3_pipeline_test.linkSystemLibrary("MLIRLinalgDialect");
        m3_pipeline_test.linkSystemLibrary("MLIRSCFDialect");
        m3_pipeline_test.linkSystemLibrary("MLIRGPUDialect");
        
        // FIX: Add the missing dialect extension libraries for BufferizableOpInterface
        m3_pipeline_test.linkSystemLibrary("MLIRFuncAllExtensions");
        m3_pipeline_test.linkSystemLibrary("MLIRTensorAllExtensions");
        
        // Add dependencies of MLIRFuncAllExtensions
        m3_pipeline_test.linkSystemLibrary("MLIRFuncInlinerExtension");
        m3_pipeline_test.linkSystemLibrary("MLIRShardingInterface");
        m3_pipeline_test.linkSystemLibrary("MLIRFuncMeshShardingExtensions");
        
        // Add mesh dialect libraries for mesh sharding symbols
        m3_pipeline_test.linkSystemLibrary("MLIRMeshDialect");
        
        // GPU-related passes and transforms
        m3_pipeline_test.linkSystemLibrary("MLIRGPUTransforms");
        m3_pipeline_test.linkSystemLibrary("MLIRGPUPipelines");
        m3_pipeline_test.linkSystemLibrary("MLIRGPUToGPURuntimeTransforms");
        
        // Linalg transforms and conversions
        m3_pipeline_test.linkSystemLibrary("MLIRLinalgTransforms");
        m3_pipeline_test.linkSystemLibrary("MLIRLinalgToStandard");
        
        // SCF transforms  
        m3_pipeline_test.linkSystemLibrary("MLIRSCFTransforms");
        
        // GPU to SPIR-V conversion
        m3_pipeline_test.linkSystemLibrary("MLIRGPUToSPIRV");
        
        // Conversion libraries needed for SPIR-V pipeline
        m3_pipeline_test.linkSystemLibrary("MLIRSCFToSPIRV");
        m3_pipeline_test.linkSystemLibrary("MLIRSCFToControlFlow");
        m3_pipeline_test.linkSystemLibrary("MLIRFuncToSPIRV");
        m3_pipeline_test.linkSystemLibrary("MLIRMemRefToSPIRV");
        m3_pipeline_test.linkSystemLibrary("MLIRVectorToSPIRV");
        m3_pipeline_test.linkSystemLibrary("MLIRArithToSPIRV");
        
        // Additional interfaces and utilities
        m3_pipeline_test.linkSystemLibrary("MLIRParallelCombiningOpInterface");
        m3_pipeline_test.linkSystemLibrary("MLIRSPIRVConversion");
        m3_pipeline_test.linkSystemLibrary("MLIRCAPIConversion");
        
        // GPU utils and interfaces
        m3_pipeline_test.linkSystemLibrary("MLIRGPUUtils");
    }
    
    if (spirv_bridge_lib != null) {
        m3_pipeline_test.linkLibrary(spirv_bridge_lib.?);
    }

    // Link Metal bridge on macOS
    if (builtin.os.tag == .macos and metal_bridge_lib != null) {
        m3_pipeline_test.linkFramework("Foundation");
        m3_pipeline_test.linkFramework("Metal");
        m3_pipeline_test.linkLibrary(metal_bridge_lib.?);
    }

    const run_m3_pipeline_test_cmd = b.addRunArtifact(m3_pipeline_test);
    run_m3_pipeline_test_cmd.step.dependOn(&m3_pipeline_test.step);

    const run_m3_pipeline_test_step = b.step("run-m3-pipeline-test", "Test complete MLIR → SPIR-V → MSL → Metal pipeline on M3");
    run_m3_pipeline_test_step.dependOn(&run_m3_pipeline_test_cmd.step);

    // Create a new Zig file for our Plan-based test
    const plan_test_src =
        \\const std = @import("std");
        \\const pcp = @import("pcp");
        \\const tensor = pcp.tensor;
        \\const ops = pcp.ops;
        \\const autodiff = pcp.autodiff;
        \\
        \\/// Helper function for pointer casting
        \\fn ptrCastHelper(comptime T: type, ptr: anytype) T {
        \\    // We still need to use alignCast for pointers that require higher alignment
        \\    return @ptrCast(@alignCast(ptr));
        \\}
        \\
        \\const Allocator = std.mem.Allocator;
        \\const Tensor = tensor.Tensor;
        \\const DType = tensor.DType;
        \\
        \\// Helper function to print tensor contents
        \\fn printTensor(t: Tensor) void {
        \\    const buf = ptrCastHelper([*]f32, t.buffer.data.ptr)[0..t.shape.elemCount()];
        \\    
        \\    std.debug.print("Shape: [", .{});
        \\    for (t.shape.dims) |dim| {
        \\        std.debug.print("{}, ", .{dim});
        \\    }
        \\    std.debug.print("]\n", .{});
        \\    
        \\    if (t.shape.rank() == 2) {
        \\        const rows = t.shape.dims[0];
        \\        const cols = t.shape.dims[1];
        \\        
        \\        for (0..rows) |i| {
        \\            std.debug.print("[ ", .{});
        \\            for (0..cols) |j| {
        \\                std.debug.print("{d:.4} ", .{buf[i * cols + j]});
        \\            }
        \\            std.debug.print("]\n", .{});
        \\        }
        \\    } else {
        \\        // For other ranks, just print all values
        \\        std.debug.print("[ ", .{});
        \\        for (buf) |val| {
        \\            std.debug.print("{d:.4} ", .{val});
        \\        }
        \\        std.debug.print("]\n", .{});
        \\    }
        \\}
        \\
        \\// Test a simple model with Plan-based approach
        \\fn testPlanBasedModel(allocator: Allocator) !void {
        \\    std.debug.print("\n=== Testing Plan-Based Model ===\n", .{});
        \\    
        \\    // Create simple tensors
        \\    var dims = [_]usize{ 2, 2 };
        \\    
        \\    // Create tensors for parameters
        \\    var w1_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
        \\    defer w1_tensor.deinit();
        \\    
        \\    try w1_tensor.setScalar(&[_]usize{0, 0}, 0.1);
        \\    try w1_tensor.setScalar(&[_]usize{0, 1}, 0.2);
        \\    try w1_tensor.setScalar(&[_]usize{1, 0}, 0.3);
        \\    try w1_tensor.setScalar(&[_]usize{1, 1}, 0.4);
        \\    
        \\    var w2_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
        \\    defer w2_tensor.deinit();
        \\    
        \\    try w2_tensor.setScalar(&[_]usize{0, 0}, 0.5);
        \\    try w2_tensor.setScalar(&[_]usize{0, 1}, 0.6);
        \\    try w2_tensor.setScalar(&[_]usize{1, 0}, 0.7);
        \\    try w2_tensor.setScalar(&[_]usize{1, 1}, 0.8);
        \\    
        \\    // Create input
        \\    var x_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
        \\    defer x_tensor.deinit();
        \\    
        \\    try x_tensor.setScalar(&[_]usize{0, 0}, 1.0);
        \\    try x_tensor.setScalar(&[_]usize{0, 1}, 2.0);
        \\    try x_tensor.setScalar(&[_]usize{1, 0}, 3.0);
        \\    try x_tensor.setScalar(&[_]usize{1, 1}, 4.0);
        \\    
        \\    // Target values
        \\    var y_tensor = try Tensor.zeros(allocator, &dims, .f32, .cpu);
        \\    defer y_tensor.deinit();
        \\    
        \\    try y_tensor.setScalar(&[_]usize{0, 0}, 0.5);
        \\    try y_tensor.setScalar(&[_]usize{0, 1}, 0.5);
        \\    try y_tensor.setScalar(&[_]usize{1, 0}, 0.5);
        \\    try y_tensor.setScalar(&[_]usize{1, 1}, 0.5);
        \\    
        \\    // Create the autodiff plans
        \\    const MatmulPlanType = autodiff.MatmulPlanWithGrad(ops.CpuBackend, f32, null, null, null);
        \\    var matmul_plan1 = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
        \\    defer matmul_plan1.deinit();
        \\    
        \\    var matmul_plan2 = autodiff.AutoDiffPlan(MatmulPlanType).init(allocator);
        \\    defer matmul_plan2.deinit();
        \\    
        \\    const ReluPlanType = autodiff.ReluPlanWithGrad(ops.CpuBackend, f32, null);
        \\    var relu_plan = autodiff.AutoDiffPlan(ReluPlanType).init(allocator);
        \\    defer relu_plan.deinit();
        \\    
        \\    // Forward pass using plans
        \\    std.debug.print("Running forward pass with plans...\n", .{});
        \\    
        \\    // h_pre = x @ w1
        \\    const h_pre = try matmul_plan1.forward(.{ .a = x_tensor, .b = w1_tensor });
        \\    defer h_pre.deinit();
        \\    
        \\    // h = relu(h_pre)
        \\    const h = try relu_plan.forward(h_pre);
        \\    defer h.deinit();
        \\    
        \\    // y_pred = h @ w2
        \\    const y_pred = try matmul_plan2.forward(.{ .a = h, .b = w2_tensor });
        \\    defer y_pred.deinit();
        \\    
        \\    // Print prediction
        \\    std.debug.print("\nPrediction (Plan-based):\n", .{});
        \\    printTensor(y_pred);
        \\    
        \\    // Compute MSE loss manually
        \\    var diff = try ops.subtract(allocator, y_pred, y_tensor);
        \\    defer diff.deinit();
        \\    
        \\    var diff_squared = try ops.multiply(allocator, diff, diff);
        \\    defer diff_squared.deinit();
        \\    
        \\    // Calculate mean manually
        \\    const diff_squared_buf = ptrCastHelper([*]f32, diff_squared.buffer.data.ptr)[0..diff_squared.shape.elemCount()];
        \\    var sum: f32 = 0.0;
        \\    for (diff_squared_buf) |val| {
        \\        sum += val;
        \\    }
        \\    const mse = sum / @as(f32, @floatFromInt(diff_squared.shape.elemCount()));
        \\    
        \\    std.debug.print("\nMSE Loss: {d:.6}\n", .{mse});
        \\    
        \\    // Create gradient with ones
        \\    var grad_ones = try Tensor.filled(allocator, y_pred.shape.dims, y_pred.dtype, 1.0, y_pred.backend);
        \\    defer grad_ones.deinit();
        \\    
        \\    // Backward pass
        \\    std.debug.print("\nRunning backward pass with plans...\n", .{});
        \\    
        \\    // Backprop through second matmul
        \\    const grads2 = try matmul_plan2.backward(grad_ones);
        \\    defer grads2.da.deinit();
        \\    defer grads2.db.deinit();
        \\    
        \\    // Backprop through relu
        \\    const relu_grad = try relu_plan.backward(grads2.da);
        \\    defer relu_grad.deinit();
        \\    
        \\    // Backprop through first matmul
        \\    const grads1 = try matmul_plan1.backward(relu_grad);
        \\    defer grads1.da.deinit();
        \\    defer grads1.db.deinit();
        \\    
        \\    // Print gradients
        \\    std.debug.print("\nGradients for w1 (Plan-based):\n", .{});
        \\    printTensor(grads1.db);
        \\    
        \\    std.debug.print("\nGradients for w2 (Plan-based):\n", .{});
        \\    printTensor(grads2.db);
        \\    
        \\    std.debug.print("\nPlan-based model test completed successfully\n", .{});
        \\}
        \\
        \\pub fn main() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\    
        \\    std.debug.print("Starting autodiff with Plan-based approach...\n", .{});
        \\    
        \\    // Test Plan-based model
        \\    try testPlanBasedModel(allocator);
        \\    
        \\    std.debug.print("\nAll tests completed successfully!\n", .{});
        \\}
    ;

    // Write the source to a file
    const plan_test_path = "src/examples/plan_based_test.zig";
    const plan_test_file = b.addWriteFile(plan_test_path, plan_test_src);

    // Create an executable from the new file
    const plan_test = b.addExecutable(.{
        .name = "plan_test",
        .root_source_file = b.path(plan_test_path),
        .target = target,
        .optimize = optimize,
    });

    // Make sure the file is written before compilation
    plan_test.step.dependOn(&plan_test_file.step);

    // Add module dependencies
    plan_test.root_module.addImport("pcp", pcp_module);
    
    // Add MLIR support to plan test
    addMLIRSupport(b, plan_test, mlir_config);
    
    if (spirv_bridge_lib != null) {
        plan_test.linkLibrary(spirv_bridge_lib.?);
    }
    

    // Install the executable
    // Install disabled - failing example
    // b.installArtifact(plan_test);

    // Run step for Plan test
    const run_plan_test_cmd = b.addRunArtifact(plan_test);
    run_plan_test_cmd.step.dependOn(&plan_test.step); // Only depend on the plan_test

    const run_plan_test_step = b.step("run-plan-test", "Run the Plan-based autodiff test");
    run_plan_test_step.dependOn(&run_plan_test_cmd.step);

    // MLIR integration test executable
    const mlir_test = b.addExecutable(.{
        .name = "mlir_test",
        .root_source_file = b.path("src/examples/mlir_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies for MLIR test
    mlir_test.root_module.addImport("pcp", pcp_module);
    
    // Add MLIR support to MLIR test
    addMLIRSupport(b, mlir_test, mlir_config);
    
    if (spirv_bridge_lib != null) {
        mlir_test.linkLibrary(spirv_bridge_lib.?);
    }
    

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
    
    // Add MLIR support to tensor test
    addMLIRSupport(b, tensor_mlir_test, mlir_config);
    
    if (spirv_bridge_lib != null) {
        tensor_mlir_test.linkLibrary(spirv_bridge_lib.?);
    }
    

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
    
    // Add MLIR support to SPIR-V test
    addMLIRSupport(b, spirv_test, mlir_config);
    
    if (spirv_bridge_lib != null) {
        spirv_test.linkLibrary(spirv_bridge_lib.?);
    }

    // Link Metal bridge on macOS
    if (builtin.os.tag == .macos) {
        spirv_test.linkFramework("Foundation");
        spirv_test.linkFramework("Metal");
        spirv_test.linkLibrary(metal_bridge_lib.?);
    }

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

    // Add module dependencies for distributed training
    main_distributed.root_module.addImport("pcp", pcp_module);
    
    // Add MLIR support to distributed training
    addMLIRSupport(b, main_distributed, mlir_config);

    // 1. Define a static C++ library for our Cap'n Proto bridge (if Cap'n Proto is available)
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
    
    if (spirv_bridge_lib != null) {
        main_distributed.linkLibrary(spirv_bridge_lib.?);
    }

    // Link Metal bridge on macOS for distributed training
    if (builtin.os.tag == .macos) {
        main_distributed.linkFramework("Foundation");
        main_distributed.linkFramework("Metal");
        main_distributed.linkLibrary(metal_bridge_lib.?);
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
