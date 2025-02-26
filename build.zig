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
}