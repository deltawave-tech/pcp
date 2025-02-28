const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the main PCP module
    const pcp_module = b.addModule("pcp", .{
        .root_source_file = b.path("src/main.zig"),
    });

    // Create the GPT-2 module 
    const gpt2_module = b.addModule("gpt2", .{
        .root_source_file = b.path("src/models/gpt2.zig"),
    });
    
    // Add dependency from GPT-2 to PCP
    gpt2_module.addImport("pcp", pcp_module);
    
    // Add unit tests
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Create a step for running the tests
    const test_step = b.step("test", "Run the unit tests");
    test_step.dependOn(&b.addRunArtifact(unit_tests).step);
    
    // GPT-2 training example executable
    const gpt2_example = b.addExecutable(.{
        .name = "gpt2_example",
        .root_source_file = b.path("src/examples/gpt2_training.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Add module dependencies for GPT-2 example
    gpt2_example.root_module.addImport("pcp", pcp_module);
    gpt2_example.root_module.addImport("gpt2", gpt2_module);
    
    // Shakespeare training example executable
    const shakespeare_example = b.addExecutable(.{
        .name = "shakespeare_example",
        .root_source_file = b.path("src/examples/shakespeare_training.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Add module dependencies for Shakespeare example
    shakespeare_example.root_module.addImport("pcp", pcp_module);
    shakespeare_example.root_module.addImport("gpt2", gpt2_module);
    
    // Install the executables
    b.installArtifact(gpt2_example);
    b.installArtifact(shakespeare_example);
    
    // Run step for GPT-2 example (default)
    const run_gpt2_cmd = b.addRunArtifact(gpt2_example);
    run_gpt2_cmd.step.dependOn(b.getInstallStep());
    
    const run_gpt2_step = b.step("run", "Run the GPT-2 training example");
    run_gpt2_step.dependOn(&run_gpt2_cmd.step);
    
    // Run step for Shakespeare example
    const run_shakespeare_cmd = b.addRunArtifact(shakespeare_example);
    run_shakespeare_cmd.step.dependOn(b.getInstallStep());
    
    const run_shakespeare_step = b.step("run-shakespeare", "Run the Shakespeare training example");
    run_shakespeare_step.dependOn(&run_shakespeare_cmd.step);
    
    // Autodiff test executable
    const autodiff_test = b.addExecutable(.{
        .name = "autodiff_test",
        .root_source_file = b.path("src/examples/autodiff_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Add module dependencies for autodiff test
    autodiff_test.root_module.addImport("pcp", pcp_module);
    
    // Install the test executable
    b.installArtifact(autodiff_test);
    
    // Run step for autodiff test
    const run_autodiff_test_cmd = b.addRunArtifact(autodiff_test);
    run_autodiff_test_cmd.step.dependOn(b.getInstallStep());
    
    const run_autodiff_test_step = b.step("run-autodiff-test", "Run the autodiff memory leak test");
    run_autodiff_test_step.dependOn(&run_autodiff_test_cmd.step);
}