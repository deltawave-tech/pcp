const std = @import("std");

pub const WorkerClass = enum {
    any,
    cpu,
    cuda,
    rocm,
    metal,
    vulkan,

    pub fn parse(raw: []const u8) ?WorkerClass {
        if (std.mem.eql(u8, raw, "any")) return .any;
        if (std.mem.eql(u8, raw, "cpu")) return .cpu;
        if (std.mem.eql(u8, raw, "cuda")) return .cuda;
        if (std.mem.eql(u8, raw, rawRocm()) or std.mem.eql(u8, raw, "hip")) return .rocm;
        if (std.mem.eql(u8, raw, "metal")) return .metal;
        if (std.mem.eql(u8, raw, "vulkan")) return .vulkan;
        return null;
    }

    pub fn asString(self: WorkerClass) []const u8 {
        return switch (self) {
            .any => "any",
            .cpu => "cpu",
            .cuda => "cuda",
            .rocm => rawRocm(),
            .metal => "metal",
            .vulkan => "vulkan",
        };
    }

    pub fn matchesBackendName(self: WorkerClass, backend_name: []const u8) bool {
        if (self == .any) return true;
        const backend_class = parse(backend_name) orelse return false;
        return backend_class == self;
    }

    fn rawRocm() []const u8 {
        return "rocm";
    }
};

pub const SchedulingPolicy = struct {
    worker_class: WorkerClass = .any,
    target_arch: ?[]const u8 = null,
    reserved_workers: usize = 0,
    max_workers: ?usize = null,

    pub fn allowsWorkerClass(self: SchedulingPolicy, worker_class: WorkerClass) bool {
        return self.worker_class == .any or self.worker_class == worker_class;
    }

    pub fn allowsWorkerTargetArch(self: SchedulingPolicy, worker_target_arch: ?[]const u8) bool {
        const required_target = self.target_arch orelse return true;
        const worker_target = worker_target_arch orelse return false;
        return std.mem.eql(u8, worker_target, required_target);
    }

    pub fn allowsWorker(self: SchedulingPolicy, worker_class: WorkerClass, worker_target_arch: ?[]const u8) bool {
        return self.allowsWorkerClass(worker_class) and self.allowsWorkerTargetArch(worker_target_arch);
    }
};
