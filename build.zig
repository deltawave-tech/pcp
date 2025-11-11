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

// Cap'n Proto configuration
const CapnpConfig = struct {
    enabled: bool = false,
    include_dir: ?[]const u8 = null,
    lib_dir: ?[]const u8 = null,
};

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

// NEW: A single config to hold all paths from the IREE build.
const IreeConfig = struct {
    enabled: bool = false,
    // Path to the root of the cloned 'iree' repo for top-level headers.
    source_dir: ?[]const u8 = null,
    // Path to the 'include' dir inside 'iree-build' for generated headers.
    build_include_dir: ?[]const u8 = null,
    // Path to the 'lib' dir inside 'iree-build' for all .a/.dylib files.
    lib_dir: ?[]const u8 = null,
};

// NEW: This is now the ONLY detection function we need for IREE/MLIR/LLVM.
fn detectIree(b: *std.Build) IreeConfig {
    var config: IreeConfig = .{};
    const iree_source_path = "../iree";
    const iree_build_path = "../iree-build";

    // Check that the core directories exist.
    std.fs.cwd().access(iree_source_path, .{}) catch return config;
    std.fs.cwd().access(iree_build_path, .{}) catch return config;

    // Construct the final paths we need for linking.
    const build_include_path = std.fs.path.join(b.allocator, &.{ iree_build_path, "include" }) catch @panic("OOM");
    const lib_path = std.fs.path.join(b.allocator, &.{ iree_build_path, "lib" }) catch @panic("OOM");

    // All checks passed.
    std.debug.print("Unified IREE/MLIR/LLVM dependencies located:\n", .{});
    std.debug.print("  Source Headers:  {s}\n", .{iree_source_path});
    std.debug.print("  Build Headers:   {s}\n", .{build_include_path});
    std.debug.print("  Libraries:       {s}\n", .{lib_path});

    config.enabled = true;
    config.source_dir = iree_source_path;
    config.build_include_dir = b.dupe(build_include_path);
    config.lib_dir = b.dupe(lib_path);
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

// NEW: Single function to link against the IREE SDK.
// This provides IREE, MLIR, StableHLO, and LLVM.
fn addIreeDependencies(target: *std.Build.Step.Compile, config: IreeConfig) void {
    if (!config.enabled) return;

    std.debug.print("==> Configuring IREE dependencies for '{s}'\n", .{target.name});

    // Add all necessary include paths.
    target.addIncludePath(.{ .cwd_relative = config.source_dir.? });
    target.addIncludePath(.{ .cwd_relative = config.build_include_dir.? });

    // Add library paths for both main lib directory and runtime subdirectory
    target.addLibraryPath(.{ .cwd_relative = config.lib_dir.? });
    // Add runtime library path
    const runtime_lib_path = std.fs.path.join(target.step.owner.allocator, &.{ config.source_dir.?, "../iree-build/runtime/src/iree/runtime" }) catch @panic("OOM");
    target.addLibraryPath(.{ .cwd_relative = runtime_lib_path });

    // Link C++ standard library.
    target.linkLibCpp();

    // Link the high-level IREE libraries. These will pull in all the
    // required MLIR, LLVM, and StableHLO dependencies automatically.
    // This REPLACES the old giant list.
    target.linkSystemLibrary("IREECompiler");
    target.linkSystemLibrary("iree_runtime_unified");

    // On macOS, we still need the system frameworks for Metal.
    if (target.root_module.resolved_target.?.result.os.tag == .macos) {
        target.linkFramework("Foundation");
        target.linkFramework("Metal");
        target.linkFramework("CoreGraphics"); // Often needed alongside Metal
    }
}

pub fn build(b: *std.Build) void {
    std.debug.print("==> Starting build script\n", .{});
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    
    // === START OF CHANGES ===

    // REMOVE all the old detection logic.
    // const mlir_config = detectMLIR(b);
    // const iree_sdk_dir = detectIree(b);

    // ADD the new unified detection.
    const iree_config = detectIree(b);

    // Keep capnp detection as it's separate.
    const capnp_config = detectCapnp(b);
    std.debug.print("==> Cap'n Proto detected: enabled={}, include={s}, lib={s}\n", .{
        capnp_config.enabled, 
        capnp_config.include_dir orelse "null", 
        capnp_config.lib_dir orelse "null"
    });

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
    
    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(dialect_test, iree_config);

    
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
    
    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(gpt2_example, iree_config);


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
    
    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(mlir_verification_test, iree_config);

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
    
    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(comptime_examples, iree_config);

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
    
    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(metal_test, iree_config);

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

    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(pass_test, iree_config);

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
    
    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(metal_benchmark, iree_config);

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

    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(m3_pipeline_test, iree_config);

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
    
    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(mlir_test, iree_config);

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
    
    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(tensor_mlir_test, iree_config);

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
    
    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(spirv_test, iree_config);

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

    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(main_distributed, iree_config);
    
    // IREE dependencies already added above with addIreeDependencies

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
    
    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(gpt2_model_test, iree_config);

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
    
    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(isolated_vjp_tests, iree_config);

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
    
    // NEW WAY: Single call for IREE dependencies
    addIreeDependencies(end_to_end_transformer_test, iree_config);

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
