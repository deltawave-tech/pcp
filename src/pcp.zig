/// Main entry point for PCP node runtimes.
/// This file handles command-line arguments and launches gateway, federation hub,
/// worker, or node-manager roles.
const std = @import("std");
const print = std.debug.print;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

// Import our distributed training components
const training_controller = @import("nodes/gateway/controllers/training_controller.zig");
const worker = @import("nodes/workers/worker.zig");
const diloco = @import("algorithms/diloco.zig");
const backend_selection = @import("backends/selection.zig");
const gateway = @import("nodes/gateway/gateway.zig");
const gateway_api = @import("nodes/gateway/api.zig");
const gateway_config = @import("nodes/gateway/config.zig");
const embedded_topology_supervisor = @import("nodes/gateway/embedded_topology_supervisor.zig");
const gateway_federation_client = @import("nodes/gateway/federation_client.zig");
const federation_hub = @import("nodes/federation_hub/hub.zig");
const federation_hub_api = @import("nodes/federation_hub/api.zig");
const shutdown = @import("runtime/shutdown.zig");
const runtime_config = @import("runtime/config.zig");
const node_manager_mod = @import("nodes/node_manager.zig");
const pcp_extensions = @import("pcp_extensions.zig");

const Worker = worker.Worker;

/// Command line arguments
const Args = struct {
    mode: Mode,
    host: []const u8,
    port: u16,
    workers: usize,
    gateway_config_path: ?[]const u8,
    backend: ?backend_selection.Backend,
    target_arch: ?[]const u8,
    supervisor_id: ?i64,
    supervise: bool,
    device_id: usize,
    scale: usize,
    api_host: []const u8,
    api_port: u16,
    control_host: []const u8,
    control_port: u16,
    api_token_env: ?[]const u8,
    extension_cli: pcp_extensions.CliArgs,
    child_args: ArrayList([]const u8),

    const Mode = enum {
        worker,
        node_manager,
        gateway,
        federation_hub,
    };

    pub fn parse(allocator: Allocator, args: [][:0]u8) !Args {
        var child_args_list = ArrayList([]const u8).init(allocator);
        errdefer child_args_list.deinit();

        if (args.len < 2) {
            return Args{
                .mode = .gateway,
                .host = "127.0.0.1",
                .port = 8080,
                .workers = 2,
                .gateway_config_path = null,
                .backend = null,
                .target_arch = null,
                .supervisor_id = null,
                .supervise = false,
                .device_id = 0,
                .scale = 1,
                .api_host = "127.0.0.1",
                .api_port = 8000,
                .control_host = "127.0.0.1",
                .control_port = 8080,
                .api_token_env = null,
                .extension_cli = .{},
                .child_args = child_args_list,
            };
        }

        var mode: Mode = .gateway;
        var host: []const u8 = "127.0.0.1";
        var port: u16 = 8080;
        var workers: usize = 2;
        var gateway_config_path: ?[]const u8 = null;
        var backend: ?backend_selection.Backend = null;
        var target_arch: ?[]const u8 = null;
        var supervisor_id: ?i64 = null;
        var supervise: bool = false;
        var device_id: usize = 0;
        var scale: usize = 1;
        var api_host: []const u8 = "127.0.0.1";
        var api_port: u16 = 8000;
        var control_host: []const u8 = "127.0.0.1";
        var control_port: u16 = 8080;
        var api_token_env: ?[]const u8 = null;
        var extension_cli = pcp_extensions.CliArgs{};

        var i: usize = 1;
        while (i < args.len) {
            if (std.mem.eql(u8, args[i], "--worker")) {
                mode = .worker;
            } else if (std.mem.eql(u8, args[i], "--shepherd")) {
                print("Error: --shepherd has been removed. Use --gateway with an embedded training or RL controller config.\n", .{});
                return error.LegacyStandaloneModeRemoved;
            } else if (std.mem.eql(u8, args[i], "--node-manager")) {
                mode = .node_manager;
            } else if (std.mem.eql(u8, args[i], "--inference")) {
                print("Error: --inference has been removed. Use --gateway with an embedded inference controller config.\n", .{});
                return error.LegacyStandaloneModeRemoved;
            } else if (std.mem.eql(u8, args[i], "--gateway")) {
                mode = .gateway;
            } else if (std.mem.eql(u8, args[i], "--federation-hub")) {
                mode = .federation_hub;
            } else if (try extension_cli.parseFlag(args, &i)) {
                // Handled by the private CLI extension.
            } else if (std.mem.eql(u8, args[i], "--config")) {
                print("Error: --config has been removed from the top-level CLI. Put controller config_path values inside --gateway-config.\n", .{});
                return error.LegacyStandaloneModeRemoved;
            } else if (std.mem.eql(u8, args[i], "--inference-config")) {
                print("Error: --inference-config has been removed from the top-level CLI. Put the embedded inference config_path inside --gateway-config.\n", .{});
                return error.LegacyStandaloneModeRemoved;
            } else if (std.mem.eql(u8, args[i], "--gateway-config")) {
                i += 1;
                if (i < args.len) {
                    gateway_config_path = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--host")) {
                i += 1;
                if (i < args.len) {
                    host = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--port")) {
                i += 1;
                if (i < args.len) {
                    port = std.fmt.parseInt(u16, args[i], 10) catch 8080;
                }
            } else if (std.mem.eql(u8, args[i], "--api-host")) {
                i += 1;
                if (i < args.len) {
                    api_host = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--api-port")) {
                i += 1;
                if (i < args.len) {
                    api_port = std.fmt.parseInt(u16, args[i], 10) catch 8000;
                }
            } else if (std.mem.eql(u8, args[i], "--gateway-host")) {
                i += 1;
                if (i < args.len) {
                    api_host = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--gateway-port")) {
                i += 1;
                if (i < args.len) {
                    api_port = std.fmt.parseInt(u16, args[i], 10) catch 18010;
                }
            } else if (std.mem.eql(u8, args[i], "--control-host")) {
                i += 1;
                if (i < args.len) {
                    control_host = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--control-port")) {
                i += 1;
                if (i < args.len) {
                    control_port = std.fmt.parseInt(u16, args[i], 10) catch 8080;
                }
            } else if (std.mem.eql(u8, args[i], "--api-token-env")) {
                i += 1;
                if (i < args.len) {
                    api_token_env = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--workers")) {
                i += 1;
                if (i < args.len) {
                    workers = std.fmt.parseInt(usize, args[i], 10) catch 2;
                }
            } else if (std.mem.eql(u8, args[i], "--connect")) {
                mode = .worker;
                i += 1;
                if (i < args.len) {
                    // Parse host:port format
                    const connect_str = args[i];
                    if (std.mem.indexOf(u8, connect_str, ":")) |colon_idx| {
                        host = connect_str[0..colon_idx];
                        port = std.fmt.parseInt(u16, connect_str[colon_idx + 1 ..], 10) catch 8080;
                    } else {
                        host = connect_str;
                    }
                }
            } else if (std.mem.eql(u8, args[i], "--model")) {
                print("Error: --model has been removed from the top-level CLI. Put model configuration inside the embedded controller config referenced by --gateway-config.\n", .{});
                return error.LegacyStandaloneModeRemoved;
            } else if (std.mem.eql(u8, args[i], "--backend")) {
                i += 1;
                if (i < args.len) {
                    const backend_str = args[i];
                    if (std.mem.eql(u8, backend_str, "cpu")) {
                        backend = .cpu;
                    } else if (std.mem.eql(u8, backend_str, "cuda")) {
                        backend = .cuda;
                    } else if (std.mem.eql(u8, backend_str, "metal")) {
                        backend = .metal;
                    } else if (std.mem.eql(u8, backend_str, "vulkan")) {
                        backend = .vulkan;
                    } else if (std.mem.eql(u8, backend_str, "rocm")) {
                        backend = .rocm;
                    } else {
                        print("Unknown backend: {s}\n", .{backend_str});
                        return error.InvalidBackend;
                    }
                }
            } else if (std.mem.eql(u8, args[i], "--target")) {
                i += 1;
                if (i < args.len) {
                    target_arch = args[i];
                }
            } else if (std.mem.eql(u8, args[i], "--supervisor-id")) {
                i += 1;
                if (i < args.len) {
                    supervisor_id = std.fmt.parseInt(i64, args[i], 10) catch null;
                }
            } else if (std.mem.eql(u8, args[i], "--device-id")) {
                i += 1;
                if (i < args.len) {
                    device_id = std.fmt.parseInt(usize, args[i], 10) catch 0;
                }
            } else if (std.mem.eql(u8, args[i], "--scale")) {
                i += 1;
                if (i < args.len) {
                    scale = std.fmt.parseInt(usize, args[i], 10) catch 1;
                }
            } else if (std.mem.eql(u8, args[i], "--resume")) {
                print("Error: --resume has been removed from the top-level CLI. Configure resume behavior inside the embedded gateway controller config.\n", .{});
                return error.LegacyStandaloneModeRemoved;
            } else if (std.mem.eql(u8, args[i], "--rl")) {
                print("Error: --rl has been removed from the top-level CLI. Enable the embedded RL controller inside --gateway-config.\n", .{});
                return error.LegacyStandaloneModeRemoved;
            } else if (std.mem.eql(u8, args[i], "--no-dashboard")) {
                print("Error: --no-dashboard has been removed; dashboard support is no longer part of PCP.\n", .{});
                return error.LegacyStandaloneModeRemoved;
            } else if (std.mem.eql(u8, args[i], "--terminate")) {
                print("Error: --terminate has been removed from the top-level CLI.\n", .{});
                return error.LegacyStandaloneModeRemoved;
            } else if (std.mem.eql(u8, args[i], "--streaming")) {
                print("Error: --streaming has been removed from the top-level CLI.\n", .{});
                return error.LegacyStandaloneModeRemoved;
            } else if (std.mem.eql(u8, args[i], "--supervise")) {
                supervise = true;
                i += 1;
                while (i < args.len) : (i += 1) {
                    try child_args_list.append(args[i]);
                }
                break;
            }
            i += 1;
        }

        return Args{
            .mode = mode,
            .host = host,
            .port = port,
            .workers = workers,
            .gateway_config_path = gateway_config_path,
            .backend = backend,
            .target_arch = target_arch,
            .supervisor_id = supervisor_id,
            .supervise = supervise,
            .device_id = device_id,
            .scale = scale,
            .api_host = api_host,
            .api_port = api_port,
            .control_host = control_host,
            .control_port = control_port,
            .api_token_env = api_token_env,
            .extension_cli = extension_cli,
            .child_args = child_args_list,
        };
    }

    pub fn deinit(self: *Args) void {
        self.child_args.deinit();
    }

    pub fn printUsage() void {
        print("Usage: pcp_distributed [options]\n", .{});
        print("Options:\n", .{});
        print("  --gateway            Run gateway node with controller subsystems and local APIs\n", .{});
        print("  --federation-hub     Run federation hub (global gateway coordination plane)\n", .{});
        print("  --worker             Run worker node and connect to a gateway worker-fabric endpoint\n", .{});
        print("  --node-manager       Run as Node Manager (spawns multiple supervised workers)\n", .{});
        print("  --supervise -- <worker_args>  Supervise one worker child with args after --\n", .{});
        print("  --gateway-config <path>  Path to gateway JSON config file\n", .{});
        pcp_extensions.printUsage();
        print("  --connect <host:port> Connect worker to gateway worker-fabric endpoint at host:port\n", .{});
        print("  --host <host>        Host to bind/connect to (default: 127.0.0.1)\n", .{});
        print("  --port <port>        Port to bind/connect to (default: 8080)\n", .{});
        print("  --api-host <host>    API host to bind for gateway/federation APIs (default: 127.0.0.1)\n", .{});
        print("  --api-port <port>    API port to bind for gateway/federation APIs (default: 8000)\n", .{});
        print("  --gateway-host <host> Gateway API host to bind (Gateway only, default: 127.0.0.1)\n", .{});
        print("  --gateway-port <port> Gateway API port to bind (Gateway only, default: 18010)\n", .{});
        print("  --control-host <host> Worker-fabric host to bind or connect (default: 127.0.0.1)\n", .{});
        print("  --control-port <port> Worker-fabric port to bind or connect (default: 8080)\n", .{});
        print("  --api-token-env <ENV> API token env var name for gateway/federation auth\n", .{});
        print("  --workers <count>    Number of workers to wait for (default: 2)\n", .{});
        print("  --backend <type>     Backend to use: cpu, cuda, metal, vulkan, rocm (default: auto)\n", .{});
        print("  --target <arch>      GPU target architecture (e.g., gfx942 for MI300X, sm_80 for A100)\n", .{});
        print("  --device-id <id>     GPU device ID to use (default: 0, for multi-GPU nodes)\n", .{});
        print("  --scale <N>          Number of supervised workers to spawn (NodeManager only, default: 1)\n", .{});
        print("  --supervisor-id <id> Internal: Supervisor ID (used by spawned workers)\n", .{});
        print("  --help               Show this help message\n", .{});
        print("\nExamples:\n", .{});
        print("  # Run gateway:\n", .{});
        print("  ./pcp --gateway --gateway-config experiments/gateway_local.json --gateway-host 127.0.0.1 --gateway-port 18010\n", .{});
        print("\n  # Run federation hub:\n", .{});
        print("  ./pcp --federation-hub --api-host 127.0.0.1 --api-port 19010\n", .{});
        print("\n  # Run a gateway node:\n", .{});
        print("  ./pcp --gateway --gateway-config gateway.json --gateway-host 127.0.0.1 --gateway-port 18010\n", .{});
        print("\n  # Run resilient worker on GPU 0:\n", .{});
        print("  ./pcp --host 127.0.0.1 --port 18080 --supervise -- --worker --connect 127.0.0.1:18080 --device-id 0\n", .{});
        print("\n  # Run 8 workers on an 8xH100 node (one per GPU):\n", .{});
        print("  for i in {{0..7}}; do ./pcp --worker --device-id $i & done\n", .{});
        print("\n  # Run node manager with 8 supervised workers:\n", .{});
        print("  ./pcp --node-manager --scale 8 --backend cuda\n", .{});
        print("\n  # Run worker against a gateway worker-fabric endpoint:\n", .{});
        print("  ./pcp --worker --connect 127.0.0.1:18080 --backend cpu\n", .{});
        pcp_extensions.printExamples();
    }
};

fn makeTestArgv(allocator: Allocator, values: []const []const u8) ![][:0]u8 {
    const argv = try allocator.alloc([:0]u8, values.len);
    var initialized: usize = 0;
    errdefer {
        for (argv[0..initialized]) |arg| allocator.free(arg);
        allocator.free(argv);
    }
    for (values, 0..) |value, i| {
        argv[i] = try allocator.dupeZ(u8, value);
        initialized += 1;
    }
    return argv;
}

fn freeTestArgv(allocator: Allocator, argv: [][:0]u8) void {
    for (argv) |arg| allocator.free(arg);
    allocator.free(argv);
}

test "CLI rejects removed standalone topology flags" {
    const cases = [_][]const []const u8{
        &.{ "pcp", "--shepherd" },
        &.{ "pcp", "--inference" },
        &.{ "pcp", "--config", "experiment.json" },
        &.{ "pcp", "--rl" },
        &.{ "pcp", "--streaming" },
    };
    for (cases) |case| {
        const argv = try makeTestArgv(std.testing.allocator, case);
        defer freeTestArgv(std.testing.allocator, argv);
        try std.testing.expectError(error.LegacyStandaloneModeRemoved, Args.parse(std.testing.allocator, argv));
    }
}

test "CLI accepts current runtime topology modes" {
    const cases = [_]struct {
        argv: []const []const u8,
        mode: Args.Mode,
    }{
        .{ .argv = &.{ "pcp", "--gateway", "--gateway-config", "gateway.json" }, .mode = .gateway },
        .{ .argv = &.{ "pcp", "--federation-hub" }, .mode = .federation_hub },
        .{ .argv = &.{ "pcp", "--worker", "--connect", "127.0.0.1:18080" }, .mode = .worker },
        .{ .argv = &.{ "pcp", "--node-manager", "--scale", "2" }, .mode = .node_manager },
    };
    for (cases) |case| {
        const argv = try makeTestArgv(std.testing.allocator, case.argv);
        defer freeTestArgv(std.testing.allocator, argv);
        var parsed = try Args.parse(std.testing.allocator, argv);
        defer parsed.deinit();
        try std.testing.expectEqual(case.mode, parsed.mode);
    }
}

test "CLI supervise mode keeps worker-fabric endpoint outside child args" {
    const argv = try makeTestArgv(std.testing.allocator, &.{
        "pcp",
        "--host",
        "127.0.0.1",
        "--port",
        "18080",
        "--supervise",
        "--",
        "--worker",
        "--connect",
        "127.0.0.1:18080",
    });
    defer freeTestArgv(std.testing.allocator, argv);

    var parsed = try Args.parse(std.testing.allocator, argv);
    defer parsed.deinit();

    try std.testing.expect(parsed.supervise);
    try std.testing.expectEqualStrings("127.0.0.1", parsed.host);
    try std.testing.expectEqual(@as(u16, 18080), parsed.port);
    try std.testing.expectEqual(@as(usize, 4), parsed.child_args.items.len);
    try std.testing.expectEqualStrings("--", parsed.child_args.items[0]);
    try std.testing.expectEqualStrings("--worker", parsed.child_args.items[1]);
}

test "stable worker ID formatter preserves Kubernetes identity components" {
    const worker_id = try formatStableWorkerId(
        std.testing.allocator,
        "site-a",
        "pcp-system",
        "gpu-node-1",
        "pcp-worker-0",
        "cuda",
        "sm_80",
        "GPU-123",
    );
    defer std.testing.allocator.free(worker_id);
    try std.testing.expectEqualStrings(
        "site-a-pcp-system-gpu-node-1-pcp-worker-0-cuda-sm_80-device-GPU-123",
        worker_id,
    );
}

test "stable worker ID components are sanitized" {
    const sanitized = try sanitizedComponent(std.testing.allocator, " node/a:0 ");
    defer std.testing.allocator.free(sanitized);
    try std.testing.expectEqualStrings("node-a-0", sanitized);

    const empty = try sanitizedComponent(std.testing.allocator, " \t ");
    defer std.testing.allocator.free(empty);
    try std.testing.expectEqualStrings("unknown", empty);
}

test "gateway readiness reports drain state" {
    const config = gateway_config.GatewayConfig{
        .gateway_id = "gateway-test",
        .lab_id = "lab-test",
    };
    var gateway_instance = gateway.Gateway.init(std.testing.allocator, config);
    defer gateway_instance.deinit();

    const initial = try gateway_instance.renderReadyJson(std.testing.allocator, false);
    defer std.testing.allocator.free(initial.body);
    try std.testing.expect(initial.ready);
    try std.testing.expect(std.mem.indexOf(u8, initial.body, "\"draining\":false") != null);

    gateway_instance.beginDrain();

    const draining = try gateway_instance.renderReadyJson(std.testing.allocator, false);
    defer std.testing.allocator.free(draining.body);
    try std.testing.expect(!draining.ready);
    try std.testing.expect(std.mem.indexOf(u8, draining.body, "\"draining\":true") != null);
}

test "gateway prometheus metrics include service and API counters" {
    const config = gateway_config.GatewayConfig{
        .gateway_id = "gateway-test",
        .lab_id = "lab-test",
    };
    var gateway_instance = gateway.Gateway.init(std.testing.allocator, config);
    defer gateway_instance.deinit();

    var registered = try gateway_instance.service_registry.register(.{
        .service_id = "training-main",
        .executor_id = "training-local",
        .service_type = "training",
        .base_url = "http://127.0.0.1:18080",
        .workers_connected = 2,
        .workers_ready = 1,
        .workers_available = 1,
        .worker_class = "gpu",
        .target_arch = "sm_80",
        .reserved_workers = 1,
        .max_workers = 2,
    });
    defer registered.deinit(std.testing.allocator);

    const body = try gateway_instance.renderPrometheusMetrics(std.testing.allocator, .{
        .requests_total = 3,
        .errors_total = 1,
        .duration_ns_total = std.time.ns_per_s,
    });
    defer std.testing.allocator.free(body);

    try std.testing.expect(std.mem.indexOf(u8, body, "pcp_gateway_info") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "pcp_gateway_service_workers") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "target_arch=\"sm_80\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "pcp_gateway_api_requests_total 3") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "pcp_gateway_api_request_duration_seconds_sum 1.000000") != null);
}

test "node-manager prometheus metrics include supervisor counts" {
    var manager = try node_manager_mod.NodeManager.init(std.testing.allocator, "127.0.0.1", 18080, "cuda", "sm_80");
    defer manager.deinit();
    manager.expected_supervisors.store(2, .release);
    manager.launched_supervisors.store(1, .release);

    const body = try manager.renderPrometheusMetrics(std.testing.allocator);
    defer std.testing.allocator.free(body);

    try std.testing.expect(std.mem.indexOf(u8, body, "pcp_node_manager_info") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "backend=\"cuda\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "target_arch=\"sm_80\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "pcp_node_manager_expected_supervisors{backend=\"cuda\",target_arch=\"sm_80\"} 2") != null);
}

fn federationHubApiThread(server: *federation_hub_api.FederationHubApiServer, host: []const u8, port: u16) !void {
    try server.start(host, port);
}

fn maybeLoadFederationHubToken(allocator: Allocator, args: Args) !?[]u8 {
    const env_name = args.api_token_env orelse if (runtime_config.isProductionMode()) "PCP_FEDERATION_HUB_TOKEN" else return null;
    return try runtime_config.loadSecretFromEnvOrFile(allocator, env_name, "PCP_FEDERATION_HUB_TOKEN_FILE", runtime_config.isProductionMode());
}

fn maybeLoadApiTokenByEnv(allocator: Allocator, env_name: ?[]const u8, canonical_file_env: ?[]const u8) !?[]u8 {
    const name = env_name orelse return null;
    return try runtime_config.loadSecretFromEnvOrFile(allocator, name, canonical_file_env, runtime_config.isProductionMode());
}

fn tokenEnvOrProductionDefault(env_name: ?[]const u8, production_default: []const u8) ?[]const u8 {
    return env_name orelse if (runtime_config.isProductionMode()) production_default else null;
}

fn envOrDefault(allocator: Allocator, name: []const u8, default_value: []const u8) ![]u8 {
    return std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => try allocator.dupe(u8, default_value),
        else => err,
    };
}

fn envOrNull(name: []const u8) ?[]const u8 {
    return std.posix.getenv(name);
}

fn firstSetRocmVisibleEnv() ?[]const u8 {
    if (envOrNull("HIP_VISIBLE_DEVICES") != null) return "HIP_VISIBLE_DEVICES";
    if (envOrNull("ROCR_VISIBLE_DEVICES") != null) return "ROCR_VISIBLE_DEVICES";
    if (envOrNull("GPU_DEVICE_ORDINAL") != null) return "GPU_DEVICE_ORDINAL";
    return null;
}

fn visibleDeviceEnvName(backend: backend_selection.Backend) ?[]const u8 {
    return switch (backend) {
        .cuda => if (envOrNull("CUDA_VISIBLE_DEVICES") != null) "CUDA_VISIBLE_DEVICES" else null,
        .rocm => firstSetRocmVisibleEnv(),
        else => null,
    };
}

fn visibleDeviceToken(backend: backend_selection.Backend, device_id: usize) ?[]const u8 {
    const env_name = visibleDeviceEnvName(backend) orelse return null;
    const raw = envOrNull(env_name) orelse return null;
    var seen: usize = 0;
    var tokens = std.mem.splitScalar(u8, raw, ',');
    while (tokens.next()) |token| {
        const trimmed = std.mem.trim(u8, token, " \t\r\n");
        if (trimmed.len == 0 or std.mem.eql(u8, trimmed, "-1")) continue;
        if (seen == device_id) return trimmed;
        seen += 1;
    }
    return null;
}

fn sanitizeWorkerIdComponent(bytes: []u8) void {
    for (bytes) |*byte| {
        const c = byte.*;
        if (std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.') continue;
        byte.* = '-';
    }
}

fn sanitizedComponent(allocator: Allocator, value: []const u8) ![]u8 {
    const trimmed = std.mem.trim(u8, value, " \t\r\n");
    const source = if (trimmed.len > 0) trimmed else "unknown";
    const out = try allocator.dupe(u8, source);
    sanitizeWorkerIdComponent(out);
    return out;
}

fn buildStableWorkerId(
    allocator: Allocator,
    backend: backend_selection.Backend,
    target_arch: ?[]const u8,
    device_id: usize,
) ![]u8 {
    if (std.process.getEnvVarOwned(allocator, "PCP_WORKER_ID")) |override| {
        defer allocator.free(override);
        return try sanitizedComponent(allocator, override);
    } else |err| switch (err) {
        error.EnvironmentVariableNotFound => {},
        else => return err,
    }

    const site_raw = try envOrDefault(allocator, "PCP_SITE_ID", "local");
    defer allocator.free(site_raw);
    const namespace_raw = try envOrDefault(allocator, "POD_NAMESPACE", "default");
    defer allocator.free(namespace_raw);
    const node_raw = try envOrDefault(allocator, "NODE_NAME", "unknown-node");
    defer allocator.free(node_raw);
    const pod_raw = try envOrDefault(allocator, "POD_NAME", "unknown-pod");
    defer allocator.free(pod_raw);

    const device_token = visibleDeviceToken(backend, device_id);
    const device_raw_owned = if (device_token) |token|
        try allocator.dupe(u8, token)
    else
        try std.fmt.allocPrint(allocator, "{d}", .{device_id});
    defer allocator.free(device_raw_owned);

    const site = try sanitizedComponent(allocator, site_raw);
    defer allocator.free(site);
    const namespace = try sanitizedComponent(allocator, namespace_raw);
    defer allocator.free(namespace);
    const node = try sanitizedComponent(allocator, node_raw);
    defer allocator.free(node);
    const pod = try sanitizedComponent(allocator, pod_raw);
    defer allocator.free(pod);
    const arch = try sanitizedComponent(allocator, target_arch orelse "default");
    defer allocator.free(arch);
    const device = try sanitizedComponent(allocator, device_raw_owned);
    defer allocator.free(device);

    return try formatStableWorkerId(allocator, site, namespace, node, pod, backend.toString(), arch, device);
}

fn formatStableWorkerId(
    allocator: Allocator,
    site: []const u8,
    namespace: []const u8,
    node: []const u8,
    pod: []const u8,
    backend: []const u8,
    arch: []const u8,
    device: []const u8,
) ![]u8 {
    return try std.fmt.allocPrint(
        allocator,
        "{s}-{s}-{s}-{s}-{s}-{s}-device-{s}",
        .{ site, namespace, node, pod, backend, arch, device },
    );
}

fn gatewayShutdownWatcher(gateway_instance: *gateway.Gateway, api_server: *gateway_api.GatewayApiServer) void {
    shutdown.waitUntilRequested();
    std.log.info("Shutdown requested; draining gateway", .{});
    gateway_instance.beginDrain();
    api_server.stop();
}

fn gatewayProbeThread(server: *gateway_api.GatewayProbeServer, host: []const u8, port: u16) !void {
    try server.start(host, port);
}

fn runGateway(allocator: Allocator, args: Args) !void {
    const config_path = args.gateway_config_path orelse {
        print("Error: --gateway-config is required for gateway mode\n", .{});
        return error.ConfigFileRequired;
    };

    var config_result = try gateway_config.loadGatewayConfig(allocator, config_path);
    defer config_result.deinit();
    var config = config_result.config;

    if (args.api_token_env) |override_env| {
        if (config.api) |*api| {
            api.token_env = override_env;
        } else {
            config.api_token_env = override_env;
        }
    }

    const api_token_env = tokenEnvOrProductionDefault(config.resolvedApiTokenEnv(), "PCP_API_TOKEN");
    const internal_api_token_env = tokenEnvOrProductionDefault(config.resolvedInternalApiTokenEnv(), "PCP_INTERNAL_TOKEN");
    const federation_token_env = tokenEnvOrProductionDefault(config.resolvedFederationTokenEnv(), "PCP_FEDERATION_HUB_TOKEN");

    const api_token = try maybeLoadApiTokenByEnv(allocator, api_token_env, "PCP_API_TOKEN_FILE");
    defer if (api_token) |token| allocator.free(token);
    const internal_api_token = try maybeLoadApiTokenByEnv(allocator, internal_api_token_env, "PCP_INTERNAL_TOKEN_FILE");
    defer if (internal_api_token) |token| allocator.free(token);
    const federation_token = if (config.resolvedFederationHubEndpoint() != null)
        try maybeLoadApiTokenByEnv(allocator, federation_token_env, "PCP_FEDERATION_HUB_TOKEN_FILE")
    else
        null;
    defer if (federation_token) |token| allocator.free(token);

    var gateway_instance = gateway.Gateway.init(allocator, config);
    defer gateway_instance.deinit();
    const worker_fabric_host = config.resolvedWorkerFabricHost(args.control_host);
    const worker_fabric_port = config.resolvedWorkerFabricPort(args.control_port);

    var api_server = gateway_api.GatewayApiServer.init(
        allocator,
        &gateway_instance,
        api_token,
        api_token_env,
        "PCP_API_TOKEN_FILE",
        internal_api_token,
        internal_api_token_env,
        "PCP_INTERNAL_TOKEN_FILE",
        federation_token,
        federation_token_env,
        "PCP_FEDERATION_HUB_TOKEN_FILE",
    );
    try api_server.listen(args.api_host, args.api_port);
    defer api_server.stop();

    const probe_port = (try runtime_config.parseOptionalPortEnv(allocator, "PCP_PROBE_PORT")) orelse 8081;
    var probe_server = gateway_api.GatewayProbeServer.init(allocator, &api_server);
    const probe_thread = try std.Thread.spawn(.{}, gatewayProbeThread, .{ &probe_server, args.api_host, probe_port });
    defer {
        probe_server.stop();
        probe_thread.join();
    }

    var federation_client: ?gateway_federation_client.FederationClient = null;
    var federation_thread: ?std.Thread = null;
    if (config.resolvedFederationHubEndpoint()) |endpoint| {
        const gateway_base_url = try std.fmt.allocPrint(allocator, "http://{s}:{d}", .{ args.api_host, args.api_port });
        defer allocator.free(gateway_base_url);
        federation_client = try gateway_federation_client.FederationClient.init(
            allocator,
            &gateway_instance,
            endpoint,
            federation_token,
            gateway_base_url,
            config.resolvedHeartbeatIntervalMs(),
        );
        federation_thread = try std.Thread.spawn(.{}, federationClientThread, .{&federation_client.?});
    }
    defer {
        if (federation_thread) |thread| {
            federation_client.?.stop();
            thread.join();
            federation_client.?.deinit();
        }
    }

    var embedded_supervisor = try embedded_topology_supervisor.EmbeddedTopologySupervisor.init(
        allocator,
        &gateway_instance,
        config,
        api_token,
        args.api_port,
        args.workers,
        worker_fabric_host,
        worker_fabric_port,
    );
    defer embedded_supervisor.deinit();
    try embedded_supervisor.start();

    const shutdown_thread = try std.Thread.spawn(.{}, gatewayShutdownWatcher, .{ &gateway_instance, &api_server });
    shutdown_thread.detach();

    print("🌉 Starting Gateway\n", .{});
    print("   Gateway ID: {s}\n", .{config.gateway_id});
    print("   Lab ID: {s}\n", .{config.lab_id});
    print("   Graph Backend: {s}\n", .{config.graph_backend});
    if (config.neo4j) |neo4j| {
        print("   Neo4j URI: {s}\n", .{neo4j.uri});
        print("   Neo4j User: {s}\n", .{neo4j.user});
        if (neo4j.database) |database| {
            print("   Neo4j Database: {s}\n", .{database});
        }
    }
    print("   API: {s}:{d}\n", .{ args.api_host, args.api_port });
    print("   Probes: {s}:{d}\n", .{ args.api_host, probe_port });
    if (internal_api_token_env) |env_name| {
        print("   Internal Event Token Env: {s}\n", .{env_name});
    }
    if (config.resolvedFederationHubEndpoint()) |endpoint| {
        print("   Federation Hub: {s}\n", .{endpoint});
    } else {
        print("   Federation Hub: not configured\n", .{});
    }
    if (config.enabledEmbeddedControllerCount() == 0) {
        print("   Embedded Controllers: disabled\n", .{});
    }
    print("   Federation: handshake enabled\n", .{});

    try api_server.runAcceptLoop();
}

fn federationClientThread(client: *gateway_federation_client.FederationClient) !void {
    try client.run();
}

fn runFederationHub(allocator: Allocator, args: Args) !void {
    var owned_api_token: ?[]u8 = null;
    defer if (owned_api_token) |token| allocator.free(token);
    owned_api_token = try maybeLoadFederationHubToken(allocator, args);

    var controller = federation_hub.FederationHub.init(allocator);
    defer controller.deinit();

    var api_server = federation_hub_api.FederationHubApiServer.init(allocator, &controller, owned_api_token);

    print("🌐 Starting Federation Hub\n", .{});
    print("   API: {s}:{d}\n", .{ args.api_host, args.api_port });

    try federationHubApiThread(&api_server, args.api_host, args.api_port);
}

fn workerShutdownWatcher(worker_instance: *Worker) void {
    shutdown.waitUntilRequested();
    std.log.info("Shutdown requested; disconnecting worker", .{});
    worker_instance.disconnect();
}

/// Run as Worker
fn runWorker(allocator: Allocator, args: Args) !void {
    print("🐉 Starting Worker...\n", .{});
    print("   Connecting to: {s}:{}\n", .{ args.host, args.port });

    const backend = args.backend orelse backend_selection.Backend.selectDefault();
    print("   Backend: {s}\n", .{backend.toString()});
    print("   Device ID: {}\n", .{args.device_id});

    if (args.target_arch) |target| {
        print("   Target Architecture: {s}\n", .{target});
    }

    const stable_worker_id = try buildStableWorkerId(allocator, backend, args.target_arch, args.device_id);
    defer allocator.free(stable_worker_id);
    print("   Worker ID: {s}\n", .{stable_worker_id});

    const worker_backend_instance = try backend_selection.createWorkerBackend(allocator, backend, args.device_id);

    const max_concurrency = try runtime_config.parsePositiveUsizeEnv(
        allocator,
        "WORKER_MAX_CONCURRENCY",
        if (runtime_config.isProductionMode()) null else 1,
        runtime_config.isProductionMode(),
    );
    print("   Max Concurrency: {}\n", .{max_concurrency});

    var worker_instance = try Worker.init(allocator, worker_backend_instance, args.supervisor_id, stable_worker_id, max_concurrency);
    defer worker_instance.deinit();

    const shutdown_thread = try std.Thread.spawn(.{}, workerShutdownWatcher, .{&worker_instance});
    shutdown_thread.detach();

    // Start worker main loop with automatic reconnection
    try worker_instance.runRobust(args.host, args.port, args.target_arch);

    print("💀 Worker shutting down\n", .{});
}

/// Run as Node Manager
fn runNodeManager(allocator: Allocator, args: Args) !void {
    const backend = args.backend orelse backend_selection.Backend.selectDefault();
    const backend_str = backend.toString();

    print("Starting Node Manager...\n", .{});
    print("   Target Backend: {s}\n", .{backend_str});
    print("   Spawning {} Supervisors\n", .{args.scale});

    const NodeManager = node_manager_mod.NodeManager;
    var manager = try NodeManager.init(
        allocator,
        args.host,
        args.port,
        backend_str,
        args.target_arch,
    );
    defer manager.deinit();

    var probe_server = node_manager_mod.NodeManagerProbeServer.init(allocator, &manager);
    const probe_thread = try std.Thread.spawn(.{}, nodeManagerProbeThread, .{ &probe_server, args.api_host, args.api_port });
    defer {
        probe_server.stop();
        probe_thread.join();
    }
    print("   Probe API: {s}:{d}\n", .{ args.api_host, args.api_port });

    try manager.spawnSupervisors(args.scale);

    const shutdown_thread = try std.Thread.spawn(.{}, nodeManagerShutdownWatcher, .{&manager});
    shutdown_thread.detach();

    // Keep running and monitor
    print("Node Manager monitoring {} supervisors...\n", .{args.scale});
    try manager.wait();

    print("Node Manager shutting down\n", .{});
}

fn nodeManagerShutdownWatcher(manager: *@import("nodes/node_manager.zig").NodeManager) void {
    shutdown.waitUntilRequested();
    std.log.info("Shutdown requested; stopping node-manager", .{});
    manager.stop();
}

fn nodeManagerProbeThread(server: *@import("nodes/node_manager.zig").NodeManagerProbeServer, host: []const u8, port: u16) !void {
    try server.start(host, port);
}

fn supervisorShutdownWatcher(supervisor: *@import("nodes/supervisor.zig").Supervisor) void {
    shutdown.waitUntilRequested();
    std.log.info("Shutdown requested; stopping supervisor", .{});
    supervisor.stop();
}

/// Main function
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const process_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, process_args);

    // Check for help
    for (process_args) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            Args.printUsage();
            return;
        }
    }

    var args = try Args.parse(allocator, process_args);
    defer args.deinit();

    if (try pcp_extensions.runIfSelected(allocator, args.extension_cli, args.backend, args.device_id)) {
        return;
    }

    try shutdown.installSignalHandlers();

    print("🪐 PCP Distributed Training System\n", .{});
    print("=====================================\n", .{});

    // Debug: Print parsed arguments
    print("Mode: {s}\n", .{@tagName(args.mode)});
    print("Host: {s}\n", .{args.host});
    print("Port: {}\n", .{args.port});
    print("Workers: {}\n", .{args.workers});
    print("=====================================\n", .{});

    // Check if supervision is enabled
    if (args.supervise) {
        print("🛡️  Running in supervision mode (resilient)\n", .{});
        print("   Child args: ", .{});
        for (args.child_args.items) |arg| {
            print("{s} ", .{arg});
        }
        print("\n", .{});

        // This process becomes the Supervisor
        // It will spawn a child process with args.child_args
        var s = try @import("nodes/supervisor.zig").Supervisor.init(
            allocator,
            args.host,
            args.port,
            args.child_args.items,
        );
        defer s.deinit();
        const shutdown_thread = try std.Thread.spawn(.{}, supervisorShutdownWatcher, .{&s});
        shutdown_thread.detach();
        try s.run();
        return;
    }

    switch (args.mode) {
        .worker => try runWorker(allocator, args),
        .node_manager => try runNodeManager(allocator, args),
        .gateway => try runGateway(allocator, args),
        .federation_hub => try runFederationHub(allocator, args),
    }
}

/// Test function for the distributed system
pub fn testDistributedSystem(allocator: Allocator) !void {
    std.log.info("Testing distributed system components...");

    // Test worker fabric controller
    try training_controller.WorkerFabricController.testWorkerFabric(allocator);

    // Test DiLoCo
    try diloco.testDiLoCo(allocator);

    std.log.info("🌚 All distributed system tests passed");
}

/// Integration test - run a mini distributed training session
pub fn testIntegration(_: Allocator) !void {
    std.log.info("Running integration test...");

    // This would spawn a gateway worker fabric and multiple workers in
    // separate threads to verify the full pipeline.

    // For now, just log that the structure is in place
    std.log.info("🌙 Integration test structure ready");
}
