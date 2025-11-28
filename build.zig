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

// NEW: IREE configuration struct
const IreeConfig = struct {
    enabled: bool = false,
    source_dir: ?[]const u8 = null, // Path to 'iree'
    build_dir: ?[]const u8 = null, // Path to 'iree-build'
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
            // Success: include_dir exists, now check lib_dir
            if (std.fs.cwd().access(lib_dir, .{})) |_| {
                // Success: both directories exist
                config.enabled = true;
                config.include_dir = b.dupe(include_dir);
                config.lib_dir = b.dupe(lib_dir);
                std.debug.print("Found Cap'n Proto via CAPNP_DIR at: {s}\n", .{capnp_dir});
                return config;
            } else |_| {
                // Failure: lib_dir does not exist
                std.debug.print("CAPNP_DIR lib directory not accessible, continuing with auto-detection\n", .{});
            }
        } else |_| {
            // Failure: include_dir does not exist
            std.debug.print("CAPNP_DIR include directory not accessible, continuing with auto-detection\n", .{});
        }
    } else |_| {
        // Environment variable not set, continue with auto-detection
    }

    // 2. Check sibling directory (e.g., ../capnproto-install)
    const pcp_root = b.build_root.path orelse ".";
    const parent = std.fs.path.dirname(pcp_root) orelse "..";
    const sibling_install = std.fs.path.join(b.allocator, &[_][]const u8{ parent, "capnproto-install" }) catch {
        std.debug.print("Failed to construct sibling Cap'n Proto path\n", .{});
        // Continue to standard paths
        return config; // Temporarily, but we'll continue below
    };
    defer b.allocator.free(sibling_install);

    const sibling_include = std.fs.path.join(b.allocator, &[_][]const u8{ sibling_install, "include" }) catch {
        std.debug.print("Failed to construct sibling include path\n", .{});
        // Continue to standard paths
        return config;
    };
    defer b.allocator.free(sibling_include);

    const sibling_lib = std.fs.path.join(b.allocator, &[_][]const u8{ sibling_install, "lib" }) catch {
        std.debug.print("Failed to construct sibling lib path\n", .{});
        // Continue to standard paths
        return config;
    };
    defer b.allocator.free(sibling_lib);

    // Check if sibling installation exists
    const sibling_header = std.fs.path.join(b.allocator, &[_][]const u8{ sibling_include, "capnp", "common.h" }) catch {
        std.debug.print("Failed to construct sibling header path\n", .{});
        // Continue to standard paths
        return config;
    };
    defer b.allocator.free(sibling_header);

    if (std.fs.cwd().access(sibling_header, .{})) |_| {
        config.enabled = true;
        config.include_dir = b.dupe(sibling_include);
        config.lib_dir = b.dupe(sibling_lib);
        std.debug.print("Found Cap'n Proto via sibling dir at: {s}\n", .{sibling_install});
        return config;
    } else |_| {
        std.debug.print("Sibling Cap'n Proto directory not found, checking standard locations\n", .{});
    }

    // 3. If no sibling directory, proceed with auto-detection of common paths
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

// NEW: Function to detect IREE installation (Refactored for Nix compatibility)
fn detectIree(b: *std.Build) IreeConfig {
    var config = IreeConfig{};
    const build_root_path = b.build_root.path orelse ".";

    // 1. Check for explicit IREE_SOURCE_DIR and IREE_BUILD_DIR
    const explicit_source = std.process.getEnvVarOwned(b.allocator, "IREE_SOURCE_DIR") catch null;
    const explicit_build = std.process.getEnvVarOwned(b.allocator, "IREE_BUILD_DIR") catch null;

    if (explicit_source != null or explicit_build != null) {
        // If either is set, both must be set
        if (explicit_source == null or explicit_build == null) {
            if (explicit_source) |s| b.allocator.free(s);
            if (explicit_build) |b_dir| b.allocator.free(b_dir);
            std.debug.print("ERROR: Both IREE_SOURCE_DIR and IREE_BUILD_DIR must be set together.\n", .{});
            return config;
        }

        const source_path = explicit_source.?;
        const build_path = explicit_build.?;

        // Verify paths exist
        if (std.fs.cwd().access(source_path, .{})) |_| {
            if (std.fs.cwd().access(build_path, .{})) |_| {
                config.enabled = true;
                config.source_dir = source_path; // Transfer ownership
                config.build_dir = build_path; // Transfer ownership
                std.debug.print("Found IREE via explicit env vars.\n  Source (IREE_SOURCE_DIR): {s}\n  Build (IREE_BUILD_DIR):   {s}\n", .{ source_path, build_path });
                return config;
            } else |_| {
                std.debug.print("ERROR: IREE_BUILD_DIR path does not exist: {s}\n", .{build_path});
            }
        } else |_| {
            std.debug.print("ERROR: IREE_SOURCE_DIR path does not exist: {s}\n", .{source_path});
        }

        // Cleanup on failure
        b.allocator.free(source_path);
        b.allocator.free(build_path);
        return config;
    }

    // 2. BACKWARD COMPATIBILITY: Check for IREE_DIR (expects iree/ and iree-build/ subdirs)
    if (std.process.getEnvVarOwned(b.allocator, "IREE_DIR") catch null) |iree_dir| {
        defer b.allocator.free(iree_dir);
        std.debug.print("Attempting to find IREE via IREE_DIR: {s}\n", .{iree_dir});

        const source_path = std.fs.path.join(b.allocator, &[_][]const u8{ iree_dir, "iree" }) catch |err| {
            std.debug.print("Failed to construct IREE source path: {s}\n", .{@errorName(err)});
            return config;
        };
        // NOTE: No defer here

        const build_path = std.fs.path.join(b.allocator, &[_][]const u8{ iree_dir, "iree-build" }) catch |err| {
            std.debug.print("Failed to construct IREE build path: {s}\n", .{@errorName(err)});
            b.allocator.free(source_path); // Clean up previous allocation
            return config;
        };
        // NOTE: No defer here

        if (std.fs.cwd().access(source_path, .{})) |_| {
            if (std.fs.cwd().access(build_path, .{})) |_| {
                config.enabled = true;
                config.source_dir = source_path; // Transfer ownership
                config.build_dir = build_path; // Transfer ownership
                std.debug.print("Found IREE via IREE_DIR.\n  Source: {s}\n  Build:  {s}\n", .{ source_path, build_path });
                // We don't free source_path/build_path here because we are giving ownership to the config struct
                return config;
            } else |_| {}
        } else |_| {}

        // If we reach here, the paths were not found, so we must clean up
        b.allocator.free(source_path);
        b.allocator.free(build_path);
        std.debug.print("IREE_DIR was set, but 'iree' or 'iree-build' not found. Falling back.\n", .{});
    }

    // 3. FALLBACK: Try relative path (sibling directories)
    const parent_dir = std.fs.path.dirname(build_root_path) orelse "..";
    const relative_source_path = std.fs.path.join(b.allocator, &[_][]const u8{ parent_dir, "iree" }) catch return config;
    const relative_build_path = std.fs.path.join(b.allocator, &[_][]const u8{ parent_dir, "iree-build" }) catch {
        b.allocator.free(relative_source_path);
        return config;
    };

    if (std.fs.cwd().access(relative_source_path, .{})) |_| {
        if (std.fs.cwd().access(relative_build_path, .{})) |_| {
            config.enabled = true;
            config.source_dir = relative_source_path; // Transfer ownership
            config.build_dir = relative_build_path; // Transfer ownership
            std.debug.print("Found IREE via relative path.\n  Source: {s}\n  Build:  {s}\n", .{ relative_source_path, relative_build_path });
            return config;
        } else |_| {}
    } else |_| {}

    // If we reach here, nothing was found. Free the memory from the relative path check.
    b.allocator.free(relative_source_path);
    b.allocator.free(relative_build_path);

    std.debug.print("IREE not found. Set IREE_SOURCE_DIR and IREE_BUILD_DIR, or IREE_DIR, or use relative path ('../iree').\n", .{});

    return config;
}

// REFACTORED: Add IREE include paths to a module
fn addIreeIncludes(mod: *std.Build.Module, b: *std.Build, iree_config: IreeConfig) void {
    if (!iree_config.enabled) return;
    const source_dir = iree_config.source_dir.?;
    const build_dir = iree_config.build_dir.?;

    // Use resolved paths from IREE detection
    mod.addIncludePath(.{ .cwd_relative = source_dir });
    mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{build_dir}) });
    mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/runtime/src", .{source_dir}) });
    // Add MLIR C API headers for @cImport auto-generation
    mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/third_party/llvm-project/mlir/include", .{source_dir}) });
    mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/llvm-project/tools/mlir/include", .{build_dir}) });
    // Add StableHLO C API headers
    mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/third_party/stablehlo", .{source_dir}) });
    mod.addIncludePath(.{ .cwd_relative = b.fmt("{s}/llvm-external-projects/stablehlo", .{build_dir}) });
    // Add our unified header directory
    mod.addIncludePath(b.path("src/mlir/include"));
}

// REFACTORED: Add IREE dependencies to a target
fn addIreeDependencies(target: *std.Build.Step.Compile, b: *std.Build, iree_config: IreeConfig) void {
    if (!iree_config.enabled) {
        std.debug.print("==> IREE not found for '{s}', skipping IREE dependencies.\n", .{target.name});
        return;
    }
    const source_dir = iree_config.source_dir.?;
    const build_dir = iree_config.build_dir.?;

    std.debug.print("==> Configuring IREE dependencies for '{s}'\n", .{target.name});

    // --- Include Paths ---
    // Add IREE include paths so @cImport can find headers
    target.addIncludePath(.{ .cwd_relative = source_dir });
    target.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{build_dir}) });
    target.addIncludePath(.{ .cwd_relative = b.fmt("{s}/runtime/src", .{source_dir}) });
    // Add MLIR C API headers for @cImport auto-generation
    target.addIncludePath(.{ .cwd_relative = b.fmt("{s}/third_party/llvm-project/mlir/include", .{source_dir}) });
    target.addIncludePath(.{ .cwd_relative = b.fmt("{s}/llvm-project/tools/mlir/include", .{build_dir}) });
    // Add StableHLO C API headers
    target.addIncludePath(.{ .cwd_relative = b.fmt("{s}/third_party/stablehlo", .{source_dir}) });
    target.addIncludePath(.{ .cwd_relative = b.fmt("{s}/llvm-external-projects/stablehlo", .{build_dir}) });
    // Add our unified header directory
    target.addIncludePath(b.path("src/mlir/include"));

    // --- Library Paths ---
    // Use resolved paths from IREE detection
    target.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{build_dir}) });
    target.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/runtime/src/iree/runtime", .{build_dir}) });
    target.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/build_tools/third_party/flatcc", .{build_dir}) });
    target.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/llvm-project/lib", .{build_dir}) });
    target.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/runtime/src/iree/base", .{build_dir}) });
    target.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/runtime/src/iree/hal/drivers/local_sync", .{build_dir}) });
    target.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/runtime/src/iree/hal/drivers/local_sync/registration", .{build_dir}) });
    target.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/runtime/src/iree/hal/drivers", .{build_dir}) });

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
    // Add StableHLO CAPI for attribute access
    dialect_anchors_lib.addCSourceFile(.{
        .file = .{ .cwd_relative = b.fmt("{s}/third_party/stablehlo/stablehlo/integrations/c/StablehloAttributes.cpp", .{source_dir}) },
        .flags = &.{"-std=c++17"},
    });

    // Add include paths needed by pass_anchors.cpp and StableHLO CAPI
    // Use resolved paths from IREE detection
    dialect_anchors_lib.addIncludePath(.{ .cwd_relative = b.fmt("{s}/third_party/llvm-project/mlir/include", .{source_dir}) });
    dialect_anchors_lib.addIncludePath(.{ .cwd_relative = b.fmt("{s}/third_party/llvm-project/llvm/include", .{source_dir}) });
    dialect_anchors_lib.addIncludePath(.{ .cwd_relative = b.fmt("{s}/llvm-project/include", .{build_dir}) });
    dialect_anchors_lib.addIncludePath(.{ .cwd_relative = b.fmt("{s}/llvm-project/tools/mlir/include", .{build_dir}) });
    dialect_anchors_lib.addIncludePath(.{ .cwd_relative = b.fmt("{s}/third_party/stablehlo", .{source_dir}) });
    dialect_anchors_lib.addIncludePath(.{ .cwd_relative = b.fmt("{s}/llvm-external-projects/stablehlo", .{build_dir}) });
    dialect_anchors_lib.linkSystemLibrary("stdc++");

    target.linkLibrary(dialect_anchors_lib);
    target.step.dependOn(&dialect_anchors_lib.step);

    // --- Link All Required Libraries ---
    target.linkSystemLibrary("stdc++");

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

    // Platform-specific GPU drivers
    if (target.root_module.resolved_target.?.result.os.tag == .macos) {
        // Metal driver for macOS (M3)
        target.linkSystemLibrary("iree_hal_drivers_metal_metal");
        target.linkSystemLibrary("iree_hal_drivers_metal_registration_registration");
        target.linkFramework("Foundation");
        target.linkFramework("Metal");
        target.linkFramework("CoreGraphics");
    } else if (target.root_module.resolved_target.?.result.os.tag == .linux) {
        // Vulkan driver for Linux
        target.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/runtime/src/iree/hal/drivers/vulkan", .{build_dir}) });
        target.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/runtime/src/iree/hal/drivers/vulkan/registration", .{build_dir}) });
        target.linkSystemLibrary("iree_hal_drivers_vulkan_vulkan");
        target.linkSystemLibrary("iree_hal_drivers_vulkan_dynamic_symbols");
        target.linkSystemLibrary("iree_hal_drivers_vulkan_registration_registration");

        // CUDA driver for Linux (NVIDIA GPUs)
        target.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/runtime/src/iree/hal/drivers/cuda", .{build_dir}) });
        target.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/runtime/src/iree/hal/drivers/cuda/registration", .{build_dir}) });
        target.linkSystemLibrary("iree_hal_drivers_cuda_cuda");
        target.linkSystemLibrary("iree_hal_drivers_cuda_dynamic_symbols");
        target.linkSystemLibrary("iree_hal_drivers_cuda_registration_registration");

        // HIP driver for Linux (AMD ROCm GPUs)
        target.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/runtime/src/iree/hal/drivers/hip", .{build_dir}) });
        target.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/runtime/src/iree/hal/drivers/hip/registration", .{build_dir}) });
        target.linkSystemLibrary("iree_hal_drivers_hip_hip");
        target.linkSystemLibrary("iree_hal_drivers_hip_dynamic_symbols");
        target.linkSystemLibrary("iree_hal_drivers_hip_registration_registration");
    }
}

// NEW REUSABLE FUNCTION for Cap'n Proto
fn addCapnpDependencies(target: *std.Build.Step.Compile, b: *std.Build, capnp_config: CapnpConfig) void {
    if (!capnp_config.enabled) {
        std.debug.print("==> Cap'n Proto not found for '{s}', skipping linkage.\n", .{target.name});
        return;
    }

    std.debug.print("==> Adding Cap'n Proto bridge for '{s}'\n", .{target.name});

    // Add include path for the executable itself to find capnp_bridge.h
    target.addIncludePath(b.path("src/network"));

    // Link against the system Cap'n Proto libraries
    if (capnp_config.lib_dir) |lib_dir| {
        target.addLibraryPath(.{ .cwd_relative = lib_dir });
    }
    target.linkSystemLibrary("capnp");
    target.linkSystemLibrary("kj");

    // Also link the C++ source files directly into the executable
    target.addCSourceFiles(.{
        .files = &.{
            "src/network/protocol.capnp.cpp",
            "src/network/capnp_bridge.cpp",
        },
        .flags = &.{"-std=c++17"},
    });

    if (capnp_config.include_dir) |include_dir| {
        target.addIncludePath(.{ .cwd_relative = include_dir });
    }
}

pub fn build(b: *std.Build) void {
    std.debug.print("==> Starting build script\n", .{});
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // === STREAMLINED BUILD LOGIC ===

    // Detect dependencies
    const capnp_config = detectCapnp(b);
    const iree_config = detectIree(b); // NEW
    std.debug.print("==> Cap'n Proto detected: enabled={}, include={s}, lib={s}\n", .{ capnp_config.enabled, capnp_config.include_dir orelse "null", capnp_config.lib_dir orelse "null" });
    std.debug.print("==> IREE detected: enabled={}, source={s}, build={s}\n", .{ iree_config.enabled, iree_config.source_dir orelse "null", iree_config.build_dir orelse "null" });

    // Create the main PCP module
    std.debug.print("==> Creating PCP module\n", .{});
    const pcp_module = b.addModule("pcp", .{
        .root_source_file = b.path("src/main.zig"),
    });
    addIreeIncludes(pcp_module, b, iree_config); // REFACTORED

    // Add network include path for @cImport in capnp_zig_wrapper.zig
    pcp_module.addIncludePath(b.path("src/network"));
    if (capnp_config.include_dir) |include_dir| {
        pcp_module.addIncludePath(.{ .cwd_relative = include_dir });
    }

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
    addIreeDependencies(m3_pipeline_test, b, iree_config); // REFACTORED

    const run_m3_pipeline_test_cmd = b.addRunArtifact(m3_pipeline_test);
    run_m3_pipeline_test_cmd.step.dependOn(&m3_pipeline_test.step);

    const run_m3_pipeline_test_step = b.step("run-m3-pipeline-test", "Test IREE Metal pipeline on M3");
    run_m3_pipeline_test_step.dependOn(&run_m3_pipeline_test_cmd.step);

    // --- NEW: CPU Pipeline Test ---
    const cpu_pipeline_test = b.addExecutable(.{
        .name = "cpu_pipeline_test",
        .root_source_file = b.path("src/examples/cpu_pipeline_test.zig"), // NEW FILE
        .target = target,
        .optimize = optimize,
    });
    cpu_pipeline_test.root_module.addImport("pcp", pcp_module);
    addIreeDependencies(cpu_pipeline_test, b, iree_config); // REFACTORED

    const run_cpu_pipeline_test_cmd = b.addRunArtifact(cpu_pipeline_test);
    run_cpu_pipeline_test_cmd.step.dependOn(&cpu_pipeline_test.step);

    const run_cpu_pipeline_test_step = b.step("run-cpu-pipeline-test", "Test IREE CPU pipeline");
    run_cpu_pipeline_test_step.dependOn(&run_cpu_pipeline_test_cmd.step);

    // --- NEW: CUDA Pipeline Test ---
    const cuda_pipeline_test = b.addExecutable(.{
        .name = "cuda_pipeline_test",
        .root_source_file = b.path("src/examples/cuda_pipeline_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    cuda_pipeline_test.root_module.addImport("pcp", pcp_module);
    addIreeDependencies(cuda_pipeline_test, b, iree_config);

    const run_cuda_pipeline_test_cmd = b.addRunArtifact(cuda_pipeline_test);
    run_cuda_pipeline_test_cmd.step.dependOn(&cuda_pipeline_test.step);

    const run_cuda_pipeline_test_step = b.step("run-cuda-pipeline-test", "Test IREE CUDA pipeline on NVIDIA GPU");
    run_cuda_pipeline_test_step.dependOn(&run_cuda_pipeline_test_cmd.step);

    // --- NEW: ROCm Pipeline Test ---
    const rocm_pipeline_test = b.addExecutable(.{
        .name = "rocm_pipeline_test",
        .root_source_file = b.path("src/examples/rocm_pipeline_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    rocm_pipeline_test.root_module.addImport("pcp", pcp_module);
    addIreeDependencies(rocm_pipeline_test, b, iree_config);

    const run_rocm_pipeline_test_cmd = b.addRunArtifact(rocm_pipeline_test);
    run_rocm_pipeline_test_cmd.step.dependOn(&rocm_pipeline_test.step);

    const run_rocm_pipeline_test_step = b.step("run-rocm-pipeline-test", "Test IREE ROCm pipeline on AMD GPU");
    run_rocm_pipeline_test_step.dependOn(&run_rocm_pipeline_test_cmd.step);

    // PCP distributed training executable
    std.debug.print("==> Creating PCP executable\n", .{});
    const pcp = b.addExecutable(.{
        .name = "pcp",
        .root_source_file = b.path("src/pcp.zig"),
        .target = target,
        .optimize = optimize,
    });
    pcp.root_module.addImport("pcp", pcp_module);

    // IREE dependencies
    addIreeDependencies(pcp, b, iree_config); // REFACTORED

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
                "src/network/protocol.capnp.cpp",
                "src/network/capnp_bridge.cpp",
            },
            .flags = &.{"-std=c++17"},
        });
        capnp_bridge_lib.linkSystemLibrary("stdc++"); // Link against libstdc++ for clang compatibility

        // NEW: Expose the public header directory to any executable that links this library.
        capnp_bridge_lib.addIncludePath(b.path("src/network"));

        // This part is for the library's own dependencies (it needs to find <capnp/c++.h>)
        if (capnp_config.include_dir) |include_dir| {
            capnp_bridge_lib.addIncludePath(.{ .cwd_relative = include_dir });
        }
        if (capnp_config.lib_dir) |lib_dir| {
            capnp_bridge_lib.addLibraryPath(.{ .cwd_relative = lib_dir });
        }
        // Note: Don't link capnp/kj here - static libraries should only contain object files.
        // The final executable (pcp) will link these libraries.

        // Add include paths for the main executable
        pcp.addIncludePath(b.path("src/network"));
        if (capnp_config.include_dir) |include_dir| {
            pcp.addIncludePath(.{ .cwd_relative = include_dir });
        }

        // 2. Link the Zig executable against our bridge and the Cap'n Proto libs
        pcp.linkLibrary(capnp_bridge_lib);
        // Ensure the bridge library is built before the main executable
        pcp.step.dependOn(&capnp_bridge_lib.step);
        // Use detected Cap'n Proto libraries
        if (capnp_config.lib_dir) |lib_dir| {
            pcp.addLibraryPath(.{ .cwd_relative = lib_dir });
        }
        pcp.linkSystemLibrary("capnp");
        pcp.linkSystemLibrary("kj");

        // Ensure libstdc++ is linked after Cap'n Proto libraries for proper symbol resolution
        pcp.linkSystemLibrary("stdc++");

        std.debug.print("==> Cap'n Proto bridge library configured successfully\n", .{});
    }

    // Install the distributed training executable
    b.installArtifact(pcp);

    // Run step for distributed training
    const run_pcp_cmd = b.addRunArtifact(pcp);
    if (b.args) |args| {
        run_pcp_cmd.addArgs(args);
    }
    run_pcp_cmd.step.dependOn(&pcp.step);

    std.debug.print("==> Registering run-distributed step\n", .{});
    const run_pcp_step = b.step("run-distributed", "Run the distributed training system");
    run_pcp_step.dependOn(&run_pcp_cmd.step);
    std.debug.print("==> run-distributed step registered successfully\n", .{});

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
    addIreeDependencies(isolated_vjp_tests, b, iree_config); // REFACTORED

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
    addIreeDependencies(mlir_optimizer_tests, b, iree_config); // REFACTORED

    // Run step for MLIR optimizer tests
    const run_mlir_optimizer_tests_cmd = b.addRunArtifact(mlir_optimizer_tests);
    run_mlir_optimizer_tests_cmd.step.dependOn(&mlir_optimizer_tests.step);

    const run_mlir_optimizer_tests_step = b.step("run-mlir-optimizer-tests", "Run MLIR optimizer numerical verification tests");
    run_mlir_optimizer_tests_step.dependOn(&run_mlir_optimizer_tests_cmd.step);
}
