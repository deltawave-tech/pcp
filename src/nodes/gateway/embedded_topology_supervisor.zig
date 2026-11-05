const std = @import("std");

const Allocator = std.mem.Allocator;
const print = std.debug.print;

const config = @import("config.zig");
const gateway = @import("gateway.zig");
const embedded_services = @import("embedded_services.zig");
const custom_extensions = @import("custom_extensions.zig");
const training_controller = @import("controllers/training_controller.zig");

const WorkerFabricController = training_controller.WorkerFabricController;

pub const EmbeddedTopologySupervisor = struct {
    allocator: Allocator,
    gateway_instance: *gateway.Gateway,
    gateway_config: config.GatewayConfig,
    api_token: ?[]const u8,
    gateway_api_port: u16,
    default_workers: usize,
    worker_fabric_host: []const u8,
    worker_fabric_port: u16,
    shared_worker_fabric: ?WorkerFabricController,
    inference: ?embedded_services.EmbeddedInferenceService,
    training: ?embedded_services.EmbeddedTrainingService,
    rl: ?embedded_services.EmbeddedRLService,
    custom_inference: ?custom_extensions.EmbeddedCustomInferenceService,
    custom_training: ?custom_extensions.EmbeddedCustomTrainingService,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        gateway_instance: *gateway.Gateway,
        gateway_config: config.GatewayConfig,
        api_token: ?[]const u8,
        gateway_api_port: u16,
        default_workers: usize,
        worker_fabric_host: []const u8,
        worker_fabric_port: u16,
    ) !Self {
        var supervisor = Self{
            .allocator = allocator,
            .gateway_instance = gateway_instance,
            .gateway_config = gateway_config,
            .api_token = api_token,
            .gateway_api_port = gateway_api_port,
            .default_workers = default_workers,
            .worker_fabric_host = worker_fabric_host,
            .worker_fabric_port = worker_fabric_port,
            .shared_worker_fabric = null,
            .inference = null,
            .training = null,
            .rl = null,
            .custom_inference = null,
            .custom_training = null,
        };

        if (gateway_config.enabledEmbeddedControllerCount() > 0) {
            supervisor.shared_worker_fabric = WorkerFabricController.init(allocator);
        }

        return supervisor;
    }

    pub fn deinit(self: *Self) void {
        if (self.inference) |*service| service.deinit();
        if (self.training) |*service| service.deinit();
        if (self.rl) |*service| service.deinit();
        if (self.custom_inference) |*service| service.deinit();
        if (self.custom_training) |*service| service.deinit();
        if (self.shared_worker_fabric) |*worker_fabric| {
            worker_fabric.stop();
            worker_fabric.deinit();
        }
    }

    pub fn start(self: *Self) !void {
        const shared_worker_fabric = if (self.shared_worker_fabric) |*worker_fabric|
            worker_fabric
        else
            return;

        if (self.gateway_config.resolvedEmbeddedInference()) |embedded_cfg| {
            self.inference = embedded_services.EmbeddedInferenceService.init(
                self.allocator,
                self.gateway_instance,
                shared_worker_fabric,
                self.api_token,
                self.gateway_api_port,
            );
            try self.inference.?.start(embedded_cfg);
            const api_host = if (embedded_cfg.api) |api_cfg| api_cfg.host orelse "127.0.0.1" else "127.0.0.1";
            const api_port = if (embedded_cfg.api) |api_cfg| api_cfg.port orelse self.gateway_api_port + 1 else self.gateway_api_port + 1;
            print("   Embedded Inference Controller: enabled\n", .{});
            print("   Worker Fabric: {s}:{d}\n", .{ self.worker_fabric_host, self.worker_fabric_port });
            print("   Embedded Inference API: {s}:{d}\n", .{ api_host, api_port });
        }

        if (self.gateway_config.resolvedEmbeddedTraining()) |embedded_cfg| {
            self.training = embedded_services.EmbeddedTrainingService.init(
                self.allocator,
                self.gateway_instance,
                shared_worker_fabric,
                self.api_token,
                self.gateway_api_port,
            );
            try self.training.?.start(embedded_cfg, self.default_workers);
            const api_host = if (embedded_cfg.api) |api_cfg| api_cfg.host orelse "127.0.0.1" else "127.0.0.1";
            const api_port = if (embedded_cfg.api) |api_cfg| api_cfg.port orelse self.gateway_api_port + 1 else self.gateway_api_port + 1;
            print("   Embedded Training Controller: enabled\n", .{});
            print("   Worker Fabric: {s}:{d}\n", .{ self.worker_fabric_host, self.worker_fabric_port });
            print("   Embedded Training API: {s}:{d}\n", .{ api_host, api_port });
        }

        if (self.gateway_config.resolvedEmbeddedRL()) |embedded_cfg| {
            self.rl = embedded_services.EmbeddedRLService.init(
                self.allocator,
                self.gateway_instance,
                shared_worker_fabric,
                self.api_token,
                self.gateway_api_port,
            );
            try self.rl.?.start(embedded_cfg, self.default_workers);
            const api_host = if (embedded_cfg.api) |api_cfg| api_cfg.host orelse "127.0.0.1" else "127.0.0.1";
            const api_port = if (embedded_cfg.api) |api_cfg| api_cfg.port orelse self.gateway_api_port + 1 else self.gateway_api_port + 1;
            print("   Embedded RL Controller: enabled\n", .{});
            print("   Worker Fabric: {s}:{d}\n", .{ self.worker_fabric_host, self.worker_fabric_port });
            print("   Embedded RL API: {s}:{d}\n", .{ api_host, api_port });
        }

        try custom_extensions.startEmbeddedServices(self, shared_worker_fabric);

        const listen_thread = try std.Thread.spawn(.{}, workerFabricListenThread, .{ shared_worker_fabric, self.worker_fabric_host, self.worker_fabric_port });
        listen_thread.detach();
    }
};

fn workerFabricListenThread(worker_fabric: *WorkerFabricController, host: []const u8, port: u16) !void {
    try worker_fabric.listen(host, port);
}
