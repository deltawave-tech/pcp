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

// Add IREE include paths to a module
fn addIreeIncludes(mod: *std.Build.Module, b: *std.Build) void {
    const absolute_workshop_path = "/Users/philipp/_projects/workshop";
    const iree_source_path = b.fmt("{s}/iree", .{absolute_workshop_path});
    const iree_build_include_path = b.fmt("{s}/iree-build/include", .{absolute_workshop_path});
    const iree_runtime_src_path = b.fmt("{s}/iree/runtime/src", .{absolute_workshop_path});
    mod.addIncludePath(.{ .cwd_relative = iree_source_path });
    mod.addIncludePath(.{ .cwd_relative = iree_build_include_path });
    mod.addIncludePath(.{ .cwd_relative = iree_runtime_src_path });
}

fn addIreeDependencies(target: *std.Build.Step.Compile, b: *std.Build) void {
    std.debug.print("==> Configuring IREE dependencies for '{s}'\n", .{target.name});
    const absolute_workshop_path = "/Users/philipp/_projects/workshop";

    // --- Path Definitions ---
    const iree_lib_path = b.fmt("{s}/iree-build/lib", .{absolute_workshop_path});
    const iree_runtime_lib_path = b.fmt("{s}/iree-build/runtime/src/iree/runtime", .{absolute_workshop_path});
    const flatcc_lib_path = b.fmt("{s}/iree-build/build_tools/third_party/flatcc", .{absolute_workshop_path});

    // --- Library Paths ---
    target.addLibraryPath(.{ .cwd_relative = iree_lib_path });
    target.addLibraryPath(.{ .cwd_relative = iree_runtime_lib_path });
    target.addLibraryPath(.{ .cwd_relative = flatcc_lib_path });
    const iree_build_lib_path = b.fmt("{s}/iree-build/llvm-project/lib", .{absolute_workshop_path});
    target.addLibraryPath(.{ .cwd_relative = iree_build_lib_path });
    const iree_base_lib_path = b.fmt("{s}/iree-build/runtime/src/iree/base", .{absolute_workshop_path});
    target.addLibraryPath(.{ .cwd_relative = iree_base_lib_path });
    const iree_hal_drivers_path = b.fmt("{s}/iree-build/runtime/src/iree/hal/drivers/local_sync", .{absolute_workshop_path});
    target.addLibraryPath(.{ .cwd_relative = iree_hal_drivers_path });
    const iree_hal_registration_path = b.fmt("{s}/iree-build/runtime/src/iree/hal/drivers/local_sync/registration", .{absolute_workshop_path});
    target.addLibraryPath(.{ .cwd_relative = iree_hal_registration_path });
    const iree_hal_drivers_base_path = b.fmt("{s}/iree-build/runtime/src/iree/hal/drivers", .{absolute_workshop_path});
    target.addLibraryPath(.{ .cwd_relative = iree_hal_drivers_base_path });
    const iree_hal_metal_path = b.fmt("{s}/iree-build/runtime/src/iree/hal/drivers/metal", .{absolute_workshop_path});
    target.addLibraryPath(.{ .cwd_relative = iree_hal_metal_path });
    const iree_hal_metal_registration_path = b.fmt("{s}/iree-build/runtime/src/iree/hal/drivers/metal/registration", .{absolute_workshop_path});
    target.addLibraryPath(.{ .cwd_relative = iree_hal_metal_registration_path });

    // --- Build and Link C++ Dialect Anchors ---
    const dialect_anchors_lib = b.addStaticLibrary(.{
        .name = "dialect_anchors",
        .target = target.root_module.resolved_target.?,
        .optimize = target.root_module.optimize.?,
    });
    dialect_anchors_lib.addCSourceFile(.{
        .file = b.path("src/mlir/pass_anchors.cpp"),
        .flags = &.{"-std=c++17"},
    });
    
    // Add include paths needed by pass_anchors.cpp
    const mlir_include_path = b.fmt("{s}/iree/third_party/llvm-project/mlir/include", .{absolute_workshop_path});
    const llvm_include_path = b.fmt("{s}/iree/third_party/llvm-project/llvm/include", .{absolute_workshop_path});
    const iree_build_include_path = b.fmt("{s}/iree-build/llvm-project/include", .{absolute_workshop_path});
    const iree_build_mlir_include_path = b.fmt("{s}/iree-build/llvm-project/tools/mlir/include", .{absolute_workshop_path});
    const stablehlo_include_path = b.fmt("{s}/iree/third_party/stablehlo", .{absolute_workshop_path});
    const stablehlo_build_include_path = b.fmt("{s}/iree-build/llvm-external-projects/stablehlo", .{absolute_workshop_path});

    dialect_anchors_lib.addIncludePath(.{ .cwd_relative = mlir_include_path });
    dialect_anchors_lib.addIncludePath(.{ .cwd_relative = llvm_include_path });
    dialect_anchors_lib.addIncludePath(.{ .cwd_relative = iree_build_include_path });
    dialect_anchors_lib.addIncludePath(.{ .cwd_relative = iree_build_mlir_include_path });
    dialect_anchors_lib.addIncludePath(.{ .cwd_relative = stablehlo_include_path });
    dialect_anchors_lib.addIncludePath(.{ .cwd_relative = stablehlo_build_include_path });
    dialect_anchors_lib.linkLibCpp();
    
    target.linkLibrary(dialect_anchors_lib);
    target.step.dependOn(&dialect_anchors_lib.step);

    // --- Link All Required Libraries ---
    target.linkLibCpp();
    
    // Core IREE
    target.linkSystemLibrary("IREECompiler");
    target.linkSystemLibrary("iree_runtime_unified");
    
    // FlatCC Dependency
    target.linkSystemLibrary("flatcc_parsing");
    
    // MLIR libraries needed for the dialect registration in pass_anchors
    target.linkSystemLibrary("MLIRFuncDialect");
    target.linkSystemLibrary("MLIRArithDialect");
    target.linkSystemLibrary("StablehloOps");
    
    // IREE HAL driver initialization (enables use_all_available_drivers)
    target.linkSystemLibrary("iree_hal_drivers_drivers");
    
    // CPU/Local-sync driver
    target.linkSystemLibrary("iree_hal_drivers_local_sync_sync_driver");
    target.linkSystemLibrary("iree_hal_drivers_local_sync_registration_registration");
    
    // Metal driver for M3
    target.linkSystemLibrary("iree_hal_drivers_metal_metal");
    target.linkSystemLibrary("iree_hal_drivers_metal_registration_registration");
    
    // macOS Frameworks
    if (target.root_module.resolved_target.?.result.os.tag == .macos) {
        target.linkFramework("Foundation");
        target.linkFramework("Metal");
        target.linkFramework("CoreGraphics");
    }
}

pub fn build(b: *std.Build) void {
    std.debug.print("==> Starting build script\n", .{});
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // === STREAMLINED BUILD LOGIC ===

    // Keep capnp detection as it's separate.
    const capnp_config = detectCapnp(b);
    std.debug.print("==> Cap'n Proto detected: enabled={}, include={s}, lib={s}\n", .{
        capnp_config.enabled,
        capnp_config.include_dir orelse "null",
        capnp_config.lib_dir orelse "null"
    });


    // Create the main PCP module
    std.debug.print("==> Creating PCP module\n", .{});
    const pcp_module = b.addModule("pcp", .{
        .root_source_file = b.path("src/main.zig"),
    });
    addIreeIncludes(pcp_module, b);
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


    // GPT-2 training example executable removed - obsolete


    // MLIR verification test removed - superseded by isolated_vjp_tests.zig





    // M3 Pipeline Test - tests complete MLIR → SPIR-V → MSL → Metal pipeline
    const m3_pipeline_test = b.addExecutable(.{
        .name = "m3_pipeline_test",
        .root_source_file = b.path("src/examples/m3_pipeline_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    m3_pipeline_test.root_module.addImport("pcp", pcp_module);

    // IREE dependencies
    addIreeDependencies(m3_pipeline_test, b);


    const run_m3_pipeline_test_cmd = b.addRunArtifact(m3_pipeline_test);
    run_m3_pipeline_test_cmd.step.dependOn(&m3_pipeline_test.step);

    const run_m3_pipeline_test_step = b.step("run-m3-pipeline-test", "Test complete MLIR → SPIR-V → MSL → Metal pipeline on M3");
    run_m3_pipeline_test_step.dependOn(&run_m3_pipeline_test_cmd.step);





    // Distributed training system executable (main_distributed.zig)
    std.debug.print("==> Creating distributed training executable\n", .{});
    const main_distributed = b.addExecutable(.{
        .name = "main_distributed",
        .root_source_file = b.path("src/main_distributed.zig"),
        .target = target,
        .optimize = optimize,
    });
    main_distributed.root_module.addImport("pcp", pcp_module);

    // IREE dependencies
    addIreeDependencies(main_distributed, b);

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

    // IREE dependencies
    addIreeDependencies(gpt2_model_test, b);


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

    // IREE dependencies
    addIreeDependencies(isolated_vjp_tests, b);


    // Install the test executable
    //b.installArtifact(isolated_vjp_tests);

    // Run step for isolated VJP tests
    const run_isolated_vjp_tests_cmd = b.addRunArtifact(isolated_vjp_tests);
    run_isolated_vjp_tests_cmd.step.dependOn(&isolated_vjp_tests.step);

    const run_isolated_vjp_tests_step = b.step("run-isolated-vjp-tests", "Run isolated VJP numerical verification tests");
    run_isolated_vjp_tests_step.dependOn(&run_isolated_vjp_tests_cmd.step);

    // MLIR Optimizer Tests - Numerical verification of Adam and Nesterov optimizers
    const mlir_optimizer_tests = b.addExecutable(.{
        .name = "mlir_optimizer_tests",
        .root_source_file = b.path("src/examples/mlir_optimizer_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module dependencies
    mlir_optimizer_tests.root_module.addImport("pcp", pcp_module);

    // IREE dependencies
    addIreeDependencies(mlir_optimizer_tests, b);

    // Run step for MLIR optimizer tests
    const run_mlir_optimizer_tests_cmd = b.addRunArtifact(mlir_optimizer_tests);
    run_mlir_optimizer_tests_cmd.step.dependOn(&mlir_optimizer_tests.step);

    const run_mlir_optimizer_tests_step = b.step("run-mlir-optimizer-tests", "Run MLIR optimizer numerical verification tests");
    run_mlir_optimizer_tests_step.dependOn(&run_mlir_optimizer_tests_cmd.step);
}
