const std = @import("std");

const model = @import("../models/federation_placement.zig");
const topology = @import("../topology.zig");

pub const Violation = enum {
    none,
    unknown_gateway,
    stale_gateway,
    incompatible_gateway,
    nondeterministic_placement,
    placement_lost,
};

pub fn checkPlacementRejection(result: model.PlacementResult) Violation {
    return switch (result.violation) {
        .none => .none,
        .unknown_gateway => .unknown_gateway,
        .stale_gateway => .stale_gateway,
        .incompatible_gateway => .incompatible_gateway,
        .invalid_request => .incompatible_gateway,
        .stale_routed_response => .placement_lost,
    };
}

pub fn checkDeterministicPlacement(left: model.PlacementResult, right: model.PlacementResult) Violation {
    if (left.decision != right.decision) return .nondeterministic_placement;
    if (left.gateway_generation != right.gateway_generation) return .nondeterministic_placement;
    if (!optionalStringEql(left.gateway_id, right.gateway_id)) return .nondeterministic_placement;
    if (!optionalStringEql(left.service_id, right.service_id)) return .nondeterministic_placement;
    return .none;
}

pub fn checkSelectedServiceMatchesRequest(
    state: model.FederationPlacementModel,
    request: model.PlacementRequest,
    result: model.PlacementResult,
) Violation {
    if (result.decision != .placed) return checkPlacementRejection(result);
    return if (state.selectedServiceMatches(request, result)) .none else .incompatible_gateway;
}

pub fn checkRoutedCompletion(result: model.PlacementResult) Violation {
    return switch (result.decision) {
        .failed_after_placement_loss => .placement_lost,
        .placed => .none,
        else => checkPlacementRejection(result),
    };
}

pub fn checkTopologyDecision(
    allocator: std.mem.Allocator,
    request_case: model.RequestCase,
    output_case: model.OutputCase,
) !void {
    const task = model.topologyTask();
    const protocol_index: usize = switch (output_case) {
        .placed => 0,
        .rejected => 1,
        .failed_after_loss => 2,
    };
    const protocol_simplexes = [_]topology.Simplex{
        task.protocol.simplexes[protocol_index],
    };
    const decisions = [_]topology.DecisionEdge{.{
        .input_index = model.inputIndex(request_case),
        .protocol_index = 0,
        .output_index = model.outputIndex(output_case),
    }};

    try topology.validateDecisionMap(
        allocator,
        .{ .simplexes = &protocol_simplexes },
        task.carrier,
        &decisions,
    );
}

fn optionalStringEql(left: ?[]const u8, right: ?[]const u8) bool {
    if (left == null or right == null) return left == null and right == null;
    return std.mem.eql(u8, left.?, right.?);
}

test "oracle maps placement rejection classes" {
    try std.testing.expectEqual(
        Violation.unknown_gateway,
        checkPlacementRejection(.{ .decision = .rejected, .violation = .unknown_gateway }),
    );
    try std.testing.expectEqual(
        Violation.stale_gateway,
        checkPlacementRejection(.{ .decision = .rejected, .violation = .stale_gateway }),
    );
    try std.testing.expectEqual(
        Violation.incompatible_gateway,
        checkPlacementRejection(.{ .decision = .rejected, .violation = .incompatible_gateway }),
    );
}

test "oracle catches nondeterministic placement" {
    try std.testing.expectEqual(
        Violation.nondeterministic_placement,
        checkDeterministicPlacement(
            .{
                .decision = .placed,
                .gateway_id = "gateway-a",
                .service_id = "training-a",
                .gateway_generation = 1,
            },
            .{
                .decision = .placed,
                .gateway_id = "gateway-b",
                .service_id = "training-b",
                .gateway_generation = 1,
            },
        ),
    );
}

test "oracle validates selected service compatibility" {
    const training_capabilities = [_][]const u8{"training.decoupled"};
    const services = [_]model.Service{
        model.makeService("training-a", "exec-a", "training", &training_capabilities, 2, "cuda", "sm_80"),
    };
    var state = model.FederationPlacementModel.init();
    try state.registerGateway("gateway-a", .connected, &services);

    const request = model.PlacementRequest{
        .selector = .{ .capability = "training.decoupled" },
        .worker_class = "cuda",
        .target_arch = "sm_80",
    };
    const result = state.place(request);

    try std.testing.expectEqual(Violation.none, checkSelectedServiceMatchesRequest(state, request, result));
    try std.testing.expectEqual(
        Violation.incompatible_gateway,
        checkSelectedServiceMatchesRequest(state, .{
            .selector = .{ .capability = "training.decoupled" },
            .worker_class = "rocm",
            .target_arch = "gfx942",
        }, result),
    );
}

test "oracle maps placement loss to stable violation" {
    try std.testing.expectEqual(
        Violation.placement_lost,
        checkRoutedCompletion(.{
            .decision = .failed_after_placement_loss,
            .violation = .stale_routed_response,
        }),
    );
}

test "oracle rejects placement outside the carrier" {
    try std.testing.expectError(
        error.DecisionOutsideCarrier,
        checkTopologyDecision(std.testing.allocator, .stale_gateway, .placed),
    );
}
