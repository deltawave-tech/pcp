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
    
    // Example executable
    const example = b.addExecutable(.{
        .name = "gpt2_example",
        .root_source_file = b.path("src/examples/gpt2_training.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Add module dependencies
    example.root_module.addImport("pcp", pcp_module);
    example.root_module.addImport("gpt2", gpt2_module);
    
    // Install the executable
    b.installArtifact(example);
    
    // Run step
    const run_cmd = b.addRunArtifact(example);
    run_cmd.step.dependOn(b.getInstallStep());
    
    const run_step = b.step("run", "Run the example");
    run_step.dependOn(&run_cmd.step);
}