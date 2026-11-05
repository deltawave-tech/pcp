const std = @import("std");

const federation_types = @import("../../../protocol/federation/types.zig");
const real_registry = @import("../../../nodes/federation_hub/gateway_registry.zig");
const topology = @import("../topology.zig");

pub const max_gateways = 6;
pub const max_services_per_gateway = 8;

pub const GatewayStatus = enum {
    connected,
    stale,
    disconnected,
};

pub const ServiceHealth = enum {
    ok,
    unknown,
    degraded,
};

pub const Decision = enum {
    idle,
    registered,
    placed,
    rejected,
    deferred,
    failed_after_placement_loss,
};

pub const Violation = enum {
    none,
    invalid_request,
    unknown_gateway,
    stale_gateway,
    incompatible_gateway,
    stale_routed_response,
};

pub const Selector = union(enum) {
    service_type: []const u8,
    capability: []const u8,
};

pub const Service = struct {
    service_id: []const u8,
    executor_id: []const u8,
    service_type: []const u8,
    health: ServiceHealth = .unknown,
    workers_available: usize = 0,
    workers_ready: usize = 0,
    worker_class: ?[]const u8 = null,
    target_arch: ?[]const u8 = null,
    capabilities: []const []const u8 = &.{},
    updated_at: i64 = 0,
};

pub const Gateway = struct {
    gateway_id: []const u8,
    status: GatewayStatus,
    generation: u32 = 1,
    services: [max_services_per_gateway]Service = undefined,
    service_count: usize = 0,

    pub fn serviceSlice(self: *const Gateway) []const Service {
        return self.services[0..self.service_count];
    }
};

pub const PlacementRequest = struct {
    selector: ?Selector = null,
    gateway_id: ?[]const u8 = null,
    service_id: ?[]const u8 = null,
    executor_id: ?[]const u8 = null,
    worker_class: ?[]const u8 = null,
    target_arch: ?[]const u8 = null,
    workers_required: ?usize = null,
};

pub const PlacementResult = struct {
    decision: Decision,
    gateway_id: ?[]const u8 = null,
    service_id: ?[]const u8 = null,
    gateway_generation: u32 = 0,
    violation: Violation = .none,
};

pub const Snapshot = struct {
    gateways_total: usize = 0,
    connected_gateways: usize = 0,
    services_total: usize = 0,
    decision: Decision = .idle,
    violation: Violation = .none,

    pub fn eql(a: Snapshot, b: Snapshot) bool {
        return a.gateways_total == b.gateways_total and
            a.connected_gateways == b.connected_gateways and
            a.services_total == b.services_total and
            a.decision == b.decision and
            a.violation == b.violation;
    }
};

pub const FederationPlacementTopologyTask = struct {
    input: topology.Complex,
    protocol: topology.Complex,
    output: topology.Complex,
    carrier: topology.CarrierMap,
    decisions: []const topology.DecisionEdge,
};

pub const RequestCase = enum {
    compatible,
    stale_gateway,
    incompatible,
};

pub const OutputCase = enum {
    placed,
    rejected,
    failed_after_loss,
};

pub const ExecutionTraceCase = enum {
    compatible_placement,
    compatible_placement_loss,
    stale_gateway_rejection,
    incompatible_gateway_rejection,
};

pub const ExecutionTraceObservation = struct {
    trace_case: ExecutionTraceCase,
    input_index: usize,
    protocol_index: usize,
    output_index: usize,
    final_decision: Decision,
    violation: Violation,
};

const compatible_input_vertices = [_]topology.Vertex{
    .{ .color = .{ .hub = 0 }, .label = .{ .symbol = "hub.registry" } },
    .{ .color = .{ .gateway = 1 }, .label = .{ .symbol = "gateway.connected.compatible" } },
    .{ .color = .{ .job = 1 }, .label = .{ .symbol = "job.requires.training" } },
};
const stale_input_vertices = [_]topology.Vertex{
    .{ .color = .{ .hub = 0 }, .label = .{ .symbol = "hub.registry" } },
    .{ .color = .{ .gateway = 1 }, .label = .{ .symbol = "gateway.stale.compatible" } },
    .{ .color = .{ .job = 1 }, .label = .{ .symbol = "job.requires.training" } },
};
const incompatible_input_vertices = [_]topology.Vertex{
    .{ .color = .{ .hub = 0 }, .label = .{ .symbol = "hub.registry" } },
    .{ .color = .{ .gateway = 1 }, .label = .{ .symbol = "gateway.connected.incompatible" } },
    .{ .color = .{ .job = 1 }, .label = .{ .symbol = "job.requires.training" } },
};

const placed_protocol_vertices = [_]topology.Vertex{
    .{ .color = .{ .hub = 0 }, .label = .{ .symbol = "placement.selected" } },
    .{ .color = .{ .gateway = 1 }, .label = .{ .symbol = "route.forwarded" } },
};
const rejected_protocol_vertices = [_]topology.Vertex{
    .{ .color = .{ .hub = 0 }, .label = .{ .symbol = "placement.rejected" } },
    .{ .color = .{ .job = 1 }, .label = .{ .symbol = "job.not_forwarded" } },
};
const failed_protocol_vertices = [_]topology.Vertex{
    .{ .color = .{ .hub = 0 }, .label = .{ .symbol = "placement.lost" } },
    .{ .color = .{ .gateway = 1 }, .label = .{ .symbol = "gateway.lost_after_route" } },
};

const placed_output_vertices = [_]topology.Vertex{
    .{ .color = .{ .hub = 0 }, .label = .{ .symbol = "output.placed" } },
    .{ .color = .{ .gateway = 1 }, .label = .{ .symbol = "gateway.compatible" } },
};
const rejected_output_vertices = [_]topology.Vertex{
    .{ .color = .{ .hub = 0 }, .label = .{ .symbol = "output.rejected" } },
    .{ .color = .{ .job = 1 }, .label = .{ .symbol = "job.no_compatible_gateway" } },
};
const failed_output_vertices = [_]topology.Vertex{
    .{ .color = .{ .hub = 0 }, .label = .{ .symbol = "output.failed_after_loss" } },
    .{ .color = .{ .gateway = 1 }, .label = .{ .symbol = "gateway.unavailable" } },
};

const input_simplexes = [_]topology.Simplex{
    .{ .vertices = &compatible_input_vertices },
    .{ .vertices = &stale_input_vertices },
    .{ .vertices = &incompatible_input_vertices },
};
const protocol_simplexes = [_]topology.Simplex{
    .{ .vertices = &placed_protocol_vertices },
    .{ .vertices = &rejected_protocol_vertices },
    .{ .vertices = &failed_protocol_vertices },
};
const output_simplexes = [_]topology.Simplex{
    .{ .vertices = &placed_output_vertices },
    .{ .vertices = &rejected_output_vertices },
    .{ .vertices = &failed_output_vertices },
};

const compatible_outputs = [_]usize{ 0, 2 };
const rejected_outputs = [_]usize{1};
const carrier_edges = [_]topology.CarrierEdge{
    .{ .input_index = 0, .output_indices = &compatible_outputs },
    .{ .input_index = 1, .output_indices = &rejected_outputs },
    .{ .input_index = 2, .output_indices = &rejected_outputs },
};
const default_decisions = [_]topology.DecisionEdge{
    .{ .input_index = 0, .protocol_index = 0, .output_index = 0 },
    .{ .input_index = 1, .protocol_index = 1, .output_index = 1 },
    .{ .input_index = 0, .protocol_index = 2, .output_index = 2 },
};
const compatible_execution_protocols = [_]usize{ 0, 2 };
const rejected_execution_protocols = [_]usize{1};
const execution_edges = [_]topology.ExecutionEdge{
    .{ .input_index = 0, .protocol_indices = &compatible_execution_protocols },
    .{ .input_index = 1, .protocol_indices = &rejected_execution_protocols },
    .{ .input_index = 2, .protocol_indices = &rejected_execution_protocols },
};
const protocol_decisions = [_]topology.ProtocolDecisionEdge{
    .{ .protocol_index = 0, .output_index = 0 },
    .{ .protocol_index = 1, .output_index = 1 },
    .{ .protocol_index = 2, .output_index = 2 },
};

pub const FederationPlacementModel = struct {
    gateways: [max_gateways]Gateway = undefined,
    gateway_count: usize = 0,
    decision: Decision = .idle,
    violation: Violation = .none,

    pub fn init() FederationPlacementModel {
        return .{};
    }

    pub fn registerGateway(
        self: *FederationPlacementModel,
        gateway_id: []const u8,
        status: GatewayStatus,
        services: []const Service,
    ) !void {
        if (services.len > max_services_per_gateway) return error.TooManyServices;
        if (self.findGatewayIndex(gateway_id)) |index| {
            var gateway = &self.gateways[index];
            gateway.status = status;
            gateway.generation += 1;
            gateway.service_count = services.len;
            @memcpy(gateway.services[0..services.len], services);
            self.decision = .registered;
            self.violation = .none;
            return;
        }

        if (self.gateway_count >= max_gateways) return error.TooManyGateways;

        var gateway = Gateway{
            .gateway_id = gateway_id,
            .status = status,
            .service_count = services.len,
        };
        @memcpy(gateway.services[0..services.len], services);
        self.gateways[self.gateway_count] = gateway;
        self.gateway_count += 1;
        self.decision = .registered;
        self.violation = .none;
    }

    pub fn heartbeat(self: *FederationPlacementModel, gateway_id: []const u8) void {
        const gateway = self.getGateway(gateway_id) orelse {
            _ = self.reject(.unknown_gateway);
            return;
        };
        gateway.status = .connected;
        self.decision = .registered;
        self.violation = .none;
    }

    pub fn markStale(self: *FederationPlacementModel, gateway_id: []const u8) void {
        const gateway = self.getGateway(gateway_id) orelse {
            _ = self.reject(.unknown_gateway);
            return;
        };
        gateway.status = .stale;
        gateway.generation += 1;
    }

    pub fn loseGateway(self: *FederationPlacementModel, gateway_id: []const u8) void {
        const gateway = self.getGateway(gateway_id) orelse {
            _ = self.reject(.unknown_gateway);
            return;
        };
        gateway.status = .disconnected;
        gateway.generation += 1;
    }

    pub fn place(self: *FederationPlacementModel, request: PlacementRequest) PlacementResult {
        if (request.selector == null) return self.reject(.invalid_request);

        if (request.gateway_id) |gateway_id| {
            const gateway = self.findGateway(gateway_id) orelse return self.reject(.unknown_gateway);
            if (gateway.status != .connected) return self.reject(.stale_gateway);
        }

        var best_gateway_index: ?usize = null;
        var best_service_index: usize = 0;

        for (self.gateways[0..self.gateway_count], 0..) |gateway, gateway_index| {
            if (request.gateway_id) |gateway_id| {
                if (!std.mem.eql(u8, gateway.gateway_id, gateway_id)) continue;
            }
            if (gateway.status != .connected) continue;

            for (gateway.serviceSlice(), 0..) |service, service_index| {
                if (!serviceMatchesRequest(service, request)) continue;
                if (best_gateway_index == null or isBetterCandidate(
                    service,
                    self.gateways[best_gateway_index.?].services[best_service_index],
                    request.workers_required,
                )) {
                    best_gateway_index = gateway_index;
                    best_service_index = service_index;
                }
            }
        }

        const gateway_index = best_gateway_index orelse return self.reject(.incompatible_gateway);
        const gateway = self.gateways[gateway_index];
        const service = gateway.services[best_service_index];
        const result = PlacementResult{
            .decision = .placed,
            .gateway_id = gateway.gateway_id,
            .service_id = service.service_id,
            .gateway_generation = gateway.generation,
        };
        self.decision = result.decision;
        self.violation = .none;
        return result;
    }

    pub fn completeRoutedPlacement(self: *FederationPlacementModel, placement: PlacementResult) PlacementResult {
        if (placement.decision != .placed or placement.gateway_id == null or placement.service_id == null) {
            return self.reject(.invalid_request);
        }

        const gateway = self.findGateway(placement.gateway_id.?) orelse return self.reject(.unknown_gateway);
        if (gateway.status != .connected or gateway.generation != placement.gateway_generation) {
            return self.failAfterPlacementLoss(.stale_routed_response);
        }

        for (gateway.serviceSlice()) |service| {
            if (std.mem.eql(u8, service.service_id, placement.service_id.?)) {
                self.decision = .placed;
                self.violation = .none;
                return placement;
            }
        }

        return self.failAfterPlacementLoss(.stale_routed_response);
    }

    pub fn selectedServiceMatches(self: FederationPlacementModel, request: PlacementRequest, result: PlacementResult) bool {
        if (result.decision != .placed) return false;
        const gateway_id = result.gateway_id orelse return false;
        const service_id = result.service_id orelse return false;
        const gateway = self.findGateway(gateway_id) orelse return false;
        if (gateway.status != .connected or gateway.generation != result.gateway_generation) return false;
        for (gateway.serviceSlice()) |service| {
            if (!std.mem.eql(u8, service.service_id, service_id)) continue;
            return serviceMatchesRequest(service, request);
        }
        return false;
    }

    pub fn snapshot(self: FederationPlacementModel) Snapshot {
        var snapshot_value = Snapshot{
            .gateways_total = self.gateway_count,
            .decision = self.decision,
            .violation = self.violation,
        };
        for (self.gateways[0..self.gateway_count]) |gateway| {
            if (gateway.status == .connected) snapshot_value.connected_gateways += 1;
            snapshot_value.services_total += gateway.service_count;
        }
        return snapshot_value;
    }

    pub fn findGatewayIndex(self: FederationPlacementModel, gateway_id: []const u8) ?usize {
        for (self.gateways[0..self.gateway_count], 0..) |gateway, index| {
            if (std.mem.eql(u8, gateway.gateway_id, gateway_id)) return index;
        }
        return null;
    }

    pub fn findGateway(self: *const FederationPlacementModel, gateway_id: []const u8) ?Gateway {
        const index = self.findGatewayIndex(gateway_id) orelse return null;
        return self.gateways[index];
    }

    fn getGateway(self: *FederationPlacementModel, gateway_id: []const u8) ?*Gateway {
        const index = self.findGatewayIndex(gateway_id) orelse return null;
        return &self.gateways[index];
    }

    fn reject(self: *FederationPlacementModel, violation: Violation) PlacementResult {
        self.decision = .rejected;
        self.violation = violation;
        return .{
            .decision = .rejected,
            .violation = violation,
        };
    }

    fn failAfterPlacementLoss(self: *FederationPlacementModel, violation: Violation) PlacementResult {
        self.decision = .failed_after_placement_loss;
        self.violation = violation;
        return .{
            .decision = .failed_after_placement_loss,
            .violation = violation,
        };
    }
};

pub fn topologyTask() FederationPlacementTopologyTask {
    const input = topology.Complex{ .simplexes = &input_simplexes };
    const output = topology.Complex{ .simplexes = &output_simplexes };
    return .{
        .input = input,
        .protocol = .{ .simplexes = &protocol_simplexes },
        .output = output,
        .carrier = .{
            .input = input,
            .output = output,
            .carries = &carrier_edges,
        },
        .decisions = &default_decisions,
    };
}

pub fn executionCarrier() topology.ExecutionCarrier {
    const task = topologyTask();
    return .{
        .input = task.input,
        .protocol = task.protocol,
        .carries = &execution_edges,
    };
}

pub fn protocolDecisions() []const topology.ProtocolDecisionEdge {
    return &protocol_decisions;
}

pub fn inputIndex(case: RequestCase) usize {
    return switch (case) {
        .compatible => 0,
        .stale_gateway => 1,
        .incompatible => 2,
    };
}

pub fn protocolIndex(case: OutputCase) usize {
    return switch (case) {
        .placed => 0,
        .rejected => 1,
        .failed_after_loss => 2,
    };
}

pub fn outputIndex(case: OutputCase) usize {
    return switch (case) {
        .placed => 0,
        .rejected => 1,
        .failed_after_loss => 2,
    };
}

pub fn runExecutionTrace(case: ExecutionTraceCase) !ExecutionTraceObservation {
    const training_capabilities = [_][]const u8{"training.decoupled"};
    const services = [_]Service{
        makeService("training-a", "exec-a", "training", &training_capabilities, 2, "cuda", "sm_80"),
    };
    const compatible_request = PlacementRequest{
        .selector = .{ .capability = "training.decoupled" },
        .worker_class = "cuda",
        .target_arch = "sm_80",
        .workers_required = 1,
    };

    var model = FederationPlacementModel.init();
    switch (case) {
        .compatible_placement => {
            try model.registerGateway("gateway-a", .connected, &services);
            const placement = model.place(compatible_request);
            const completed = model.completeRoutedPlacement(placement);
            return observeExecutionTrace(case, .compatible, completed);
        },
        .compatible_placement_loss => {
            try model.registerGateway("gateway-a", .connected, &services);
            const placement = model.place(compatible_request);
            model.loseGateway("gateway-a");
            const completed = model.completeRoutedPlacement(placement);
            return observeExecutionTrace(case, .compatible, completed);
        },
        .stale_gateway_rejection => {
            try model.registerGateway("gateway-a", .stale, &services);
            const rejected = model.place(.{
                .selector = .{ .service_type = "training" },
                .gateway_id = "gateway-a",
            });
            return observeExecutionTrace(case, .stale_gateway, rejected);
        },
        .incompatible_gateway_rejection => {
            try model.registerGateway("gateway-a", .connected, &services);
            const rejected = model.place(.{
                .selector = .{ .capability = "training.decoupled" },
                .worker_class = "rocm",
                .target_arch = "gfx942",
            });
            return observeExecutionTrace(case, .incompatible, rejected);
        },
    }
}

pub fn makeService(
    service_id: []const u8,
    executor_id: []const u8,
    service_type: []const u8,
    capabilities: []const []const u8,
    workers_available: usize,
    worker_class: ?[]const u8,
    target_arch: ?[]const u8,
) Service {
    return .{
        .service_id = service_id,
        .executor_id = executor_id,
        .service_type = service_type,
        .health = .ok,
        .workers_available = workers_available,
        .workers_ready = workers_available,
        .worker_class = worker_class,
        .target_arch = target_arch,
        .capabilities = capabilities,
    };
}

fn serviceMatchesRequest(service: Service, request: PlacementRequest) bool {
    const selector = request.selector orelse return false;
    switch (selector) {
        .service_type => |service_type| {
            if (!std.mem.eql(u8, service.service_type, service_type)) return false;
        },
        .capability => |capability| {
            if (!serviceHasCapability(service, capability)) return false;
        },
    }
    if (request.service_id) |service_id| {
        if (!std.mem.eql(u8, service.service_id, service_id)) return false;
    }
    if (request.executor_id) |executor_id| {
        if (!std.mem.eql(u8, service.executor_id, executor_id)) return false;
    }
    if (request.worker_class) |worker_class| {
        const value = service.worker_class orelse return false;
        if (!std.mem.eql(u8, value, worker_class)) return false;
    }
    if (request.target_arch) |target_arch| {
        const value = service.target_arch orelse return false;
        if (!std.mem.eql(u8, value, target_arch)) return false;
    }
    return true;
}

fn serviceHasCapability(service: Service, required_capability: []const u8) bool {
    for (service.capabilities) |capability| {
        if (std.mem.eql(u8, capability, required_capability)) return true;
    }
    return false;
}

fn isBetterCandidate(candidate: Service, current: Service, workers_required: ?usize) bool {
    const candidate_ok = candidate.health == .ok;
    const current_ok = current.health == .ok;
    if (candidate_ok != current_ok) return candidate_ok;

    if (workers_required) |required| {
        const candidate_sufficient = candidate.workers_available >= required;
        const current_sufficient = current.workers_available >= required;
        if (candidate_sufficient != current_sufficient) return candidate_sufficient;
    }

    if (candidate.workers_available != current.workers_available) {
        return candidate.workers_available > current.workers_available;
    }

    if (candidate.workers_ready != current.workers_ready) {
        return candidate.workers_ready > current.workers_ready;
    }

    return candidate.updated_at > current.updated_at;
}

fn observeExecutionTrace(
    trace_case: ExecutionTraceCase,
    request_case: RequestCase,
    result: PlacementResult,
) ExecutionTraceObservation {
    const output_case: OutputCase = switch (result.decision) {
        .placed => .placed,
        .failed_after_placement_loss => .failed_after_loss,
        else => .rejected,
    };
    return .{
        .trace_case = trace_case,
        .input_index = inputIndex(request_case),
        .protocol_index = protocolIndex(output_case),
        .output_index = outputIndex(output_case),
        .final_decision = result.decision,
        .violation = result.violation,
    };
}

test "federation placement topology task validates default decisions" {
    const task = topologyTask();
    try topology.validateDecisionMap(std.testing.allocator, task.protocol, task.carrier, task.decisions);
}

test "federation execution carrier composes generated traces with task carrier" {
    const task = topologyTask();
    const execution = executionCarrier();
    try topology.validateExecutionDecisionMap(std.testing.allocator, execution, task.carrier, protocolDecisions());

    const cases = [_]ExecutionTraceCase{
        .compatible_placement,
        .compatible_placement_loss,
        .stale_gateway_rejection,
        .incompatible_gateway_rejection,
    };
    var observations: [cases.len]topology.ExecutionObservation = undefined;
    for (cases, 0..) |case, index| {
        const observed = try runExecutionTrace(case);
        try std.testing.expect(execution.protocolCarriedByInput(observed.input_index, observed.protocol_index));
        try std.testing.expect(task.carrier.outputCarriedByInput(observed.input_index, observed.output_index));
        observations[index] = .{
            .input_index = observed.input_index,
            .protocol_index = observed.protocol_index,
        };
    }

    try topology.validateExecutionObservations(std.testing.allocator, execution, &observations);
}

test "model places compatible connected gateway deterministically" {
    const training_capabilities = [_][]const u8{"training.decoupled"};
    const services_a = [_]Service{
        makeService("training-a", "exec-a", "training", &training_capabilities, 1, "cuda", "sm_80"),
    };
    const services_b = [_]Service{
        makeService("training-b", "exec-b", "training", &training_capabilities, 4, "cuda", "sm_80"),
    };
    var left = FederationPlacementModel.init();
    var right = FederationPlacementModel.init();
    try left.registerGateway("gateway-a", .connected, &services_a);
    try left.registerGateway("gateway-b", .connected, &services_b);
    try right.registerGateway("gateway-a", .connected, &services_a);
    try right.registerGateway("gateway-b", .connected, &services_b);

    const request = PlacementRequest{
        .selector = .{ .capability = "training.decoupled" },
        .worker_class = "cuda",
        .target_arch = "sm_80",
        .workers_required = 2,
    };
    const left_result = left.place(request);
    const right_result = right.place(request);

    try std.testing.expectEqual(Decision.placed, left_result.decision);
    try std.testing.expectEqualStrings("gateway-b", left_result.gateway_id.?);
    try std.testing.expectEqualStrings("training-b", left_result.service_id.?);
    try std.testing.expectEqualStrings(left_result.gateway_id.?, right_result.gateway_id.?);
    try std.testing.expectEqualStrings(left_result.service_id.?, right_result.service_id.?);
}

test "model rejects requested unknown stale and incompatible gateways" {
    const training_capabilities = [_][]const u8{"training.decoupled"};
    const services = [_]Service{
        makeService("training-a", "exec-a", "training", &training_capabilities, 2, "cuda", "sm_80"),
    };
    const request = PlacementRequest{
        .selector = .{ .capability = "training.decoupled" },
        .gateway_id = "gateway-missing",
        .worker_class = "cuda",
        .target_arch = "sm_80",
    };
    var model = FederationPlacementModel.init();
    try model.registerGateway("gateway-a", .connected, &services);

    const unknown_result = model.place(request);
    try std.testing.expectEqual(Violation.unknown_gateway, unknown_result.violation);

    model.markStale("gateway-a");
    const stale_result = model.place(.{
        .selector = .{ .capability = "training.decoupled" },
        .gateway_id = "gateway-a",
        .worker_class = "cuda",
        .target_arch = "sm_80",
    });
    try std.testing.expectEqual(Violation.stale_gateway, stale_result.violation);

    try model.registerGateway("gateway-a", .connected, &services);
    const incompatible_result = model.place(.{
        .selector = .{ .capability = "training.decoupled" },
        .worker_class = "rocm",
        .target_arch = "gfx942",
    });
    try std.testing.expectEqual(Violation.incompatible_gateway, incompatible_result.violation);
}

test "gateway loss and re-registration invalidate old routed placements" {
    const training_capabilities = [_][]const u8{"training.decoupled"};
    const services = [_]Service{
        makeService("training-a", "exec-a", "training", &training_capabilities, 2, "cuda", "sm_80"),
    };
    var model = FederationPlacementModel.init();
    try model.registerGateway("gateway-a", .connected, &services);

    const placement = model.place(.{
        .selector = .{ .service_type = "training" },
        .gateway_id = "gateway-a",
    });
    try std.testing.expectEqual(Decision.placed, placement.decision);

    model.loseGateway("gateway-a");
    const lost_result = model.completeRoutedPlacement(placement);
    try std.testing.expectEqual(Decision.failed_after_placement_loss, lost_result.decision);
    try std.testing.expectEqual(Violation.stale_routed_response, lost_result.violation);

    try model.registerGateway("gateway-a", .connected, &services);
    const stale_generation_result = model.completeRoutedPlacement(placement);
    try std.testing.expectEqual(Decision.failed_after_placement_loss, stale_generation_result.decision);
    try std.testing.expectEqual(Violation.stale_routed_response, stale_generation_result.violation);
}

test "model selection matches current gateway registry behavior" {
    const allocator = std.testing.allocator;
    const training_capabilities = [_][]const u8{"training.decoupled"};
    const ads_a = [_]federation_types.ServiceAdvertisement{.{
        .service_id = "training-a",
        .executor_id = "exec-a",
        .service_type = "training",
        .base_url = "http://gateway-a/training",
        .health_status = "ok",
        .worker_count = 1,
        .ready_worker_count = 1,
        .workers_available = 1,
        .worker_class = "cuda",
        .target_arch = "sm_80",
        .capabilities = &training_capabilities,
        .updated_at = 1,
    }};
    const ads_b = [_]federation_types.ServiceAdvertisement{.{
        .service_id = "training-b",
        .executor_id = "exec-b",
        .service_type = "training",
        .base_url = "http://gateway-b/training",
        .health_status = "ok",
        .worker_count = 4,
        .ready_worker_count = 4,
        .workers_available = 4,
        .worker_class = "cuda",
        .target_arch = "sm_80",
        .capabilities = &training_capabilities,
        .updated_at = 2,
    }};

    var registry = real_registry.GatewayRegistry.init(allocator);
    defer registry.deinit();
    var registered_a = try registry.upsert(.{
        .gateway_id = "gateway-a",
        .lab_id = "lab-a",
        .base_url = "http://gateway-a",
        .graph_backend = "mesh",
        .services = &ads_a,
    });
    defer registered_a.deinit(allocator);
    var registered_b = try registry.upsert(.{
        .gateway_id = "gateway-b",
        .lab_id = "lab-b",
        .base_url = "http://gateway-b",
        .graph_backend = "mesh",
        .services = &ads_b,
    });
    defer registered_b.deinit(allocator);

    var model = FederationPlacementModel.init();
    const services_a = [_]Service{
        makeService("training-a", "exec-a", "training", &training_capabilities, 1, "cuda", "sm_80"),
    };
    const services_b = [_]Service{
        makeService("training-b", "exec-b", "training", &training_capabilities, 4, "cuda", "sm_80"),
    };
    try model.registerGateway("gateway-a", .connected, &services_a);
    try model.registerGateway("gateway-b", .connected, &services_b);

    const placement = federation_types.PlacementRequest{
        .worker_class = "cuda",
        .target_arch = "sm_80",
        .workers_required = 2,
    };
    var selected = (try registry.selectServiceByCapability(allocator, "training.decoupled", placement)).?;
    defer selected.deinit(allocator);

    const model_result = model.place(.{
        .selector = .{ .capability = "training.decoupled" },
        .worker_class = "cuda",
        .target_arch = "sm_80",
        .workers_required = 2,
    });

    try std.testing.expectEqual(Decision.placed, model_result.decision);
    try std.testing.expectEqualStrings(selected.gateway_id, model_result.gateway_id.?);
    try std.testing.expectEqualStrings(selected.service.service_id, model_result.service_id.?);
}

test "model mirrors registry workers required ranking behavior" {
    const allocator = std.testing.allocator;
    const training_capabilities = [_][]const u8{"training.decoupled"};
    const ads = [_]federation_types.ServiceAdvertisement{.{
        .service_id = "training-a",
        .executor_id = "exec-a",
        .service_type = "training",
        .base_url = "http://gateway-a/training",
        .health_status = "ok",
        .worker_count = 1,
        .ready_worker_count = 1,
        .workers_available = 1,
        .worker_class = "cuda",
        .target_arch = "sm_80",
        .capabilities = &training_capabilities,
    }};

    var registry = real_registry.GatewayRegistry.init(allocator);
    defer registry.deinit();
    var registered = try registry.upsert(.{
        .gateway_id = "gateway-a",
        .lab_id = "lab-a",
        .base_url = "http://gateway-a",
        .graph_backend = "mesh",
        .services = &ads,
    });
    defer registered.deinit(allocator);

    var model = FederationPlacementModel.init();
    const services = [_]Service{
        makeService("training-a", "exec-a", "training", &training_capabilities, 1, "cuda", "sm_80"),
    };
    try model.registerGateway("gateway-a", .connected, &services);

    const placement = federation_types.PlacementRequest{
        .worker_class = "cuda",
        .target_arch = "sm_80",
        .workers_required = 2,
    };
    var selected = (try registry.selectServiceByCapability(allocator, "training.decoupled", placement)).?;
    defer selected.deinit(allocator);

    const result = model.place(.{
        .selector = .{ .capability = "training.decoupled" },
        .worker_class = "cuda",
        .target_arch = "sm_80",
        .workers_required = 2,
    });

    try std.testing.expectEqual(Decision.placed, result.decision);
    try std.testing.expectEqualStrings(selected.gateway_id, result.gateway_id.?);
    try std.testing.expectEqualStrings(selected.service.service_id, result.service_id.?);
}
