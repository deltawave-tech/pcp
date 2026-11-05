const std = @import("std");

const tasks = @import("tasks.zig");
const topology = @import("topology.zig");

pub const VertexRef = struct {
    color: topology.Color,
    label: []const u8,
};

pub const ProjectionSpec = struct {
    source: VertexRef,
    target: VertexRef,
};

pub const OwnedProjection = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    domain: topology.Complex,
    codomain: topology.Complex,
    entries: []topology.VertexMapEntry,

    pub fn vertexMap(self: *const OwnedProjection) topology.VertexMap {
        return .{
            .domain = self.domain,
            .codomain = self.codomain,
            .entries = self.entries,
        };
    }

    pub fn deinit(self: *OwnedProjection) void {
        self.allocator.free(self.entries);
        self.entries = &.{};
    }
};

pub const ProjectionCarrierLink = struct {
    name: []const u8,
    projection: topology.VertexMap,
    next: topology.SimplicialCarrierMap,
};

pub const OwnedGlobalTrainingComposition = struct {
    allocator: std.mem.Allocator,
    workload: tasks.OwnedProtocolTask,
    federation: tasks.OwnedProtocolTask,
    scheduling: tasks.OwnedProtocolTask,
    reservation: tasks.OwnedProtocolTask,
    fragment: tasks.OwnedProtocolTask,
    rda: tasks.OwnedProtocolTask,
    global: tasks.OwnedProtocolTask,
    workload_to_federation: OwnedProjection,
    federation_to_scheduling: OwnedProjection,
    scheduling_to_reservation: OwnedProjection,
    reservation_to_fragment: OwnedProjection,
    fragment_to_rda: OwnedProjection,
    rda_to_global: OwnedProjection,
    carrier: topology.OwnedSimplicialCarrierMap,
    abstract_global_carrier: topology.OwnedSimplicialCarrierMap,

    pub fn composedCarrier(self: *const OwnedGlobalTrainingComposition) topology.SimplicialCarrierMap {
        return self.carrier.carrierMap();
    }

    pub fn abstractCarrier(self: *const OwnedGlobalTrainingComposition) topology.SimplicialCarrierMap {
        return self.abstract_global_carrier.carrierMap();
    }

    pub fn deinit(self: *OwnedGlobalTrainingComposition) void {
        self.abstract_global_carrier.deinit();
        self.carrier.deinit();
        self.rda_to_global.deinit();
        self.fragment_to_rda.deinit();
        self.reservation_to_fragment.deinit();
        self.scheduling_to_reservation.deinit();
        self.federation_to_scheduling.deinit();
        self.workload_to_federation.deinit();
        self.global.deinit();
        self.rda.deinit();
        self.fragment.deinit();
        self.reservation.deinit();
        self.scheduling.deinit();
        self.federation.deinit();
        self.workload.deinit();
        self.* = undefined;
    }
};

pub fn buildProjectionFromSpecs(
    allocator: std.mem.Allocator,
    name: []const u8,
    domain: topology.Complex,
    codomain: topology.Complex,
    specs: []const ProjectionSpec,
) !OwnedProjection {
    var entries = std.ArrayList(topology.VertexMapEntry).init(allocator);
    errdefer entries.deinit();

    for (specs) |spec| {
        const source = try findVertexByRef(allocator, domain, spec.source);
        const target = try findVertexByRef(allocator, codomain, spec.target);
        try entries.append(.{ .source = source, .target = target });
    }

    const owned_entries = try entries.toOwnedSlice();
    errdefer allocator.free(owned_entries);

    const projection = OwnedProjection{
        .allocator = allocator,
        .name = name,
        .domain = domain,
        .codomain = codomain,
        .entries = owned_entries,
    };
    try projection.vertexMap().validate(allocator, .{});
    return projection;
}

pub fn composeProjectedCarriers(
    allocator: std.mem.Allocator,
    start: topology.SimplicialCarrierMap,
    links: []const ProjectionCarrierLink,
) !topology.OwnedSimplicialCarrierMap {
    if (links.len == 0) return error.EmptyComposition;

    var current_map = start;
    var current_owned: ?topology.OwnedSimplicialCarrierMap = null;
    errdefer if (current_owned) |*owned| owned.deinit();

    for (links) |link| {
        var projected = try current_map.composeWithMap(allocator, link.projection);
        defer projected.deinit();

        if (current_owned) |*owned| {
            owned.deinit();
            current_owned = null;
        }

        current_owned = try projected.carrierMap().composeWithCarrier(allocator, link.next);
        current_map = current_owned.?.carrierMap();
    }

    return current_owned orelse error.EmptyComposition;
}

pub fn validateProjectedRefinement(
    allocator: std.mem.Allocator,
    detailed: topology.SimplicialCarrierMap,
    final_projection: topology.VertexMap,
    abstract_task: topology.SimplicialCarrierMap,
) !void {
    var projected = try detailed.composeWithMap(allocator, final_projection);
    defer projected.deinit();
    try validateCarrierRefines(allocator, projected.carrierMap(), abstract_task);
}

pub fn buildGlobalTrainingComposition(
    allocator: std.mem.Allocator,
) !OwnedGlobalTrainingComposition {
    var workload = try tasks.buildWorkloadNormalizationTask(allocator);
    errdefer workload.deinit();
    var federation = try tasks.buildFederationPlacementTask(allocator);
    errdefer federation.deinit();
    var scheduling = try tasks.buildGatewayLeaseSchedulingTask(allocator);
    errdefer scheduling.deinit();
    var reservation = try tasks.buildReservationLifecycleTask(allocator);
    errdefer reservation.deinit();
    var fragment = try tasks.buildDecoupledFragmentTask(allocator);
    errdefer fragment.deinit();
    var rda = try tasks.buildRdaMergeTask(allocator);
    errdefer rda.deinit();
    var global = try tasks.buildMultiGatewayGlobalJobTask(allocator);
    errdefer global.deinit();

    const workload_task = workload.task();
    const federation_task = federation.task();
    const scheduling_task = scheduling.task();
    const reservation_task = reservation.task();
    const fragment_task = fragment.task();
    const rda_task = rda.task();
    const global_task = global.task();

    var workload_to_federation = try buildWorkloadToFederationProjection(allocator, workload_task, federation_task);
    errdefer workload_to_federation.deinit();
    var federation_to_scheduling = try buildFederationToSchedulingProjection(allocator, federation_task, scheduling_task);
    errdefer federation_to_scheduling.deinit();
    var scheduling_to_reservation = try buildSchedulingToReservationProjection(allocator, scheduling_task, reservation_task);
    errdefer scheduling_to_reservation.deinit();
    var reservation_to_fragment = try buildReservationToFragmentProjection(allocator, reservation_task, fragment_task);
    errdefer reservation_to_fragment.deinit();
    var fragment_to_rda = try buildFragmentToRdaProjection(allocator, fragment_task, rda_task);
    errdefer fragment_to_rda.deinit();
    var rda_to_global = try buildRdaToGlobalProjection(allocator, rda_task, global_task);
    errdefer rda_to_global.deinit();

    const links = [_]ProjectionCarrierLink{
        .{
            .name = "workload -> federation",
            .projection = workload_to_federation.vertexMap(),
            .next = federation_task.carrier,
        },
        .{
            .name = "federation -> scheduling",
            .projection = federation_to_scheduling.vertexMap(),
            .next = scheduling_task.carrier,
        },
        .{
            .name = "scheduling -> reservation",
            .projection = scheduling_to_reservation.vertexMap(),
            .next = reservation_task.carrier,
        },
        .{
            .name = "reservation -> fragment",
            .projection = reservation_to_fragment.vertexMap(),
            .next = fragment_task.carrier,
        },
        .{
            .name = "fragment -> RDA",
            .projection = fragment_to_rda.vertexMap(),
            .next = rda_task.carrier,
        },
    };

    var carrier = try composeProjectedCarriers(allocator, workload_task.carrier, &links);
    errdefer carrier.deinit();

    var abstract_global_carrier = try buildWorkloadGlobalAbstractCarrier(
        allocator,
        workload_task.input,
        global_task.output,
    );
    errdefer abstract_global_carrier.deinit();

    try validateProjectedRefinement(
        allocator,
        carrier.carrierMap(),
        rda_to_global.vertexMap(),
        abstract_global_carrier.carrierMap(),
    );

    return .{
        .allocator = allocator,
        .workload = workload,
        .federation = federation,
        .scheduling = scheduling,
        .reservation = reservation,
        .fragment = fragment,
        .rda = rda,
        .global = global,
        .workload_to_federation = workload_to_federation,
        .federation_to_scheduling = federation_to_scheduling,
        .scheduling_to_reservation = scheduling_to_reservation,
        .reservation_to_fragment = reservation_to_fragment,
        .fragment_to_rda = fragment_to_rda,
        .rda_to_global = rda_to_global,
        .carrier = carrier,
        .abstract_global_carrier = abstract_global_carrier,
    };
}

fn buildWorkloadToFederationProjection(
    allocator: std.mem.Allocator,
    workload: tasks.ProtocolTask,
    federation: tasks.ProtocolTask,
) !OwnedProjection {
    const specs = [_]ProjectionSpec{
        p(.{ .api_caller = 0 }, "status.accepted", .{ .hub = 0 }, "registry.current"),
        p(.{ .gateway = 0 }, "route.regular_training", .{ .gateway = 1 }, "gateway.connected.compatible"),
        p(.{ .job = 1 }, "job.normalized.regular", .{ .job = 30 }, "job.requires.training"),
        p(.{ .api_caller = 0 }, "status.rejected", .{ .hub = 0 }, "registry.current.incompatible"),
        p(.{ .gateway = 0 }, "route.rejected", .{ .gateway = 1 }, "gateway.connected.incompatible"),
        p(.{ .job = 1 }, "job.rejected", .{ .job = 32 }, "job.requires.training"),
    };
    return buildProjectionFromSpecs(allocator, "workload -> federation", workload.output, federation.input, &specs);
}

fn buildFederationToSchedulingProjection(
    allocator: std.mem.Allocator,
    federation: tasks.ProtocolTask,
    scheduling: tasks.ProtocolTask,
) !OwnedProjection {
    const specs = [_]ProjectionSpec{
        p(.{ .hub = 0 }, "placement.placed", .{ .gateway = 0 }, "pool.idle_capacity"),
        p(.{ .gateway = 1 }, "gateway.route.accepted", .{ .worker = 0 }, "worker.cuda.idle.available"),
        p(.{ .job = 30 }, "job.forwarded", .{ .job = 10 }, "lease.training.request"),
        p(.{ .hub = 0 }, "placement.failed_after_loss", .{ .gateway = 0 }, "policy.inference.reserved_workers"),
        p(.{ .gateway = 1 }, "gateway.route_lost", .{ .worker = 0 }, "worker.cuda.last_inference_capacity"),
        p(.{ .job = 30 }, "job.failed_retryable", .{ .job = 12 }, "lease.training.request"),
        p(.{ .hub = 0 }, "placement.rejected.stale", .{ .gateway = 0 }, "policy.training.max_workers_reached"),
        p(.{ .hub = 0 }, "placement.rejected.incompatible", .{ .gateway = 0 }, "policy.training.max_workers_reached"),
        p(.{ .gateway = 1 }, "gateway.not_selected", .{ .worker = 0 }, "worker.cuda.idle.maxed"),
        p(.{ .job = 31 }, "job.rejected", .{ .job = 11 }, "lease.training.request"),
        p(.{ .job = 32 }, "job.rejected", .{ .job = 11 }, "lease.training.request"),
    };
    return buildProjectionFromSpecs(allocator, "federation -> scheduling", federation.output, scheduling.input, &specs);
}

fn buildSchedulingToReservationProjection(
    allocator: std.mem.Allocator,
    scheduling: tasks.ProtocolTask,
    reservation: tasks.ProtocolTask,
) !OwnedProjection {
    const specs = [_]ProjectionSpec{
        p(.{ .gateway = 0 }, "lease.reserved", .{ .gateway = 0 }, "reservation.active"),
        p(.{ .job = 10 }, "reservation.active", .{ .job = 20 }, "reservation.commit"),
        p(.{ .worker = 0 }, "worker.leased.training", .{ .worker = 0 }, "worker.reserved.training"),
        p(.{ .gateway = 0 }, "lease.rejected.max_workers", .{ .gateway = 0 }, "reservation.active.expire"),
        p(.{ .job = 11 }, "reservation.rejected", .{ .job = 22 }, "reservation.expire"),
        p(.{ .worker = 0 }, "worker.unleased", .{ .worker = 0 }, "worker.reserved.training.expire"),
        p(.{ .gateway = 0 }, "lease.deferred.preserve_capacity", .{ .gateway = 0 }, "reservation.active.release"),
        p(.{ .job = 12 }, "reservation.deferred", .{ .job = 21 }, "reservation.release"),
        p(.{ .worker = 0 }, "worker.reserved_for_inference", .{ .worker = 0 }, "worker.reserved.training.release"),
    };
    return buildProjectionFromSpecs(allocator, "scheduling -> reservation", scheduling.output, reservation.input, &specs);
}

fn buildReservationToFragmentProjection(
    allocator: std.mem.Allocator,
    reservation: tasks.ProtocolTask,
    fragment: tasks.ProtocolTask,
) !OwnedProjection {
    const specs = [_]ProjectionSpec{
        p(.{ .gateway = 0 }, "reservation.committed", .{ .gateway = 0 }, "window.open.quorum"),
        p(.{ .job = 20 }, "job.queued_reserved", .{ .session = 40 }, "session.active"),
        p(.{ .worker = 0 }, "worker.leased.training", .{ .worker = 0 }, "fragment.fresh.unique_sender.quorum"),
        p(.{ .gateway = 0 }, "reservation.released", .{ .gateway = 0 }, "window.canceled"),
        p(.{ .gateway = 0 }, "reservation.expired", .{ .gateway = 0 }, "window.canceled"),
        p(.{ .gateway = 0 }, "reservation.canceled.cleanup_complete", .{ .gateway = 0 }, "window.canceled"),
        p(.{ .job = 21 }, "job.not_queued", .{ .session = 45 }, "session.canceled"),
        p(.{ .job = 22 }, "job.not_queued", .{ .session = 45 }, "session.canceled"),
        p(.{ .job = 23 }, "job.canceled", .{ .session = 45 }, "session.canceled"),
        p(.{ .worker = 0 }, "worker.idle", .{ .worker = 0 }, "fragment.post_cancel"),
    };
    return buildProjectionFromSpecs(allocator, "reservation -> fragment", reservation.output, fragment.input, &specs);
}

fn buildFragmentToRdaProjection(
    allocator: std.mem.Allocator,
    fragment: tasks.ProtocolTask,
    rda: tasks.ProtocolTask,
) !OwnedProjection {
    const specs = [_]ProjectionSpec{
        p(.{ .gateway = 0 }, "merge.committed", .{ .gateway = 0 }, "merge_kind.rda.direction"),
        p(.{ .session = 40 }, "session.progressed", .{ .session = 50 }, "window.quorum"),
        p(.{ .worker = 0 }, "update.accepted", .{ .worker = 0 }, "participants.direction_nonzero"),
        p(.{ .gateway = 0 }, "merge.deferred", .{ .gateway = 0 }, "merge_kind.rda.below_quorum"),
        p(.{ .gateway = 0 }, "merge.not_committed", .{ .gateway = 0 }, "merge_kind.rda.below_quorum"),
        p(.{ .gateway = 0 }, "merge.canceled", .{ .gateway = 0 }, "merge_kind.rda.below_quorum"),
        p(.{ .session = 41 }, "session.waiting", .{ .session = 52 }, "window.below_quorum"),
        p(.{ .session = 42 }, "session.active", .{ .session = 52 }, "window.below_quorum"),
        p(.{ .session = 43 }, "session.active", .{ .session = 52 }, "window.below_quorum"),
        p(.{ .session = 44 }, "session.active", .{ .session = 52 }, "window.below_quorum"),
        p(.{ .session = 45 }, "session.canceled", .{ .session = 52 }, "window.below_quorum"),
        p(.{ .worker = 0 }, "update.pending", .{ .worker = 0 }, "participants.insufficient"),
        p(.{ .worker = 0 }, "update.rejected.duplicate", .{ .worker = 0 }, "participants.insufficient"),
        p(.{ .worker = 0 }, "update.rejected.context", .{ .worker = 0 }, "participants.insufficient"),
        p(.{ .worker = 0 }, "update.rejected.vector_clock", .{ .worker = 0 }, "participants.insufficient"),
        p(.{ .worker = 0 }, "update.rejected.post_cancel", .{ .worker = 0 }, "participants.insufficient"),
    };
    return buildProjectionFromSpecs(allocator, "fragment -> RDA", fragment.output, rda.input, &specs);
}

fn buildRdaToGlobalProjection(
    allocator: std.mem.Allocator,
    rda: tasks.ProtocolTask,
    global: tasks.ProtocolTask,
) !OwnedProjection {
    const specs = [_]ProjectionSpec{
        p(.{ .gateway = 0 }, "rda.output.direction_norm_carried", .{ .hub = 0 }, "global_job.committed"),
        p(.{ .session = 50 }, "merge.committed", .{ .job = 60 }, "job.running"),
        p(.{ .worker = 0 }, "participants.used", .{ .gateway = 1 }, "gateway_a.committed"),
        p(.{ .gateway = 0 }, "rda.output.zero", .{ .hub = 0 }, "global_job.deferred"),
        p(.{ .session = 51 }, "merge.committed", .{ .job = 63 }, "job.queued"),
        p(.{ .worker = 0 }, "participants.canceling_direction", .{ .gateway = 1 }, "gateway_a.hold_or_release"),
        p(.{ .gateway = 0 }, "rda.output.none", .{ .hub = 0 }, "global_job.deferred"),
        p(.{ .session = 52 }, "merge.deferred", .{ .job = 63 }, "job.queued"),
        p(.{ .worker = 0 }, "participants.insufficient", .{ .gateway = 2 }, "gateway_b.await_capacity"),
    };
    return buildProjectionFromSpecs(allocator, "RDA -> global outcome", rda.output, global.output, &specs);
}

fn buildWorkloadGlobalAbstractCarrier(
    allocator: std.mem.Allocator,
    input: topology.Complex,
    output: topology.Complex,
) !topology.OwnedSimplicialCarrierMap {
    var regular = try simplexFromRefs(allocator, input, &.{
        vr(.{ .api_caller = 0 }, "request.regular"),
        vr(.{ .gateway = 0 }, "parser.training.regular"),
        vr(.{ .job = 1 }, "job.training.regular"),
    });
    defer regular.deinit();
    var invalid = try simplexFromRefs(allocator, input, &.{
        vr(.{ .api_caller = 0 }, "request.invalid"),
        vr(.{ .gateway = 0 }, "parser.training.invalid"),
        vr(.{ .job = 1 }, "job.training.invalid"),
    });
    defer invalid.deinit();
    var committed = try simplexFromRefs(allocator, output, &.{
        vr(.{ .hub = 0 }, "global_job.committed"),
        vr(.{ .job = 60 }, "job.running"),
        vr(.{ .gateway = 1 }, "gateway_a.committed"),
        vr(.{ .gateway = 2 }, "gateway_b.committed"),
    });
    defer committed.deinit();
    var deferred = try simplexFromRefs(allocator, output, &.{
        vr(.{ .hub = 0 }, "global_job.deferred"),
        vr(.{ .job = 63 }, "job.queued"),
        vr(.{ .gateway = 1 }, "gateway_a.hold_or_release"),
        vr(.{ .gateway = 2 }, "gateway_b.await_capacity"),
    });
    defer deferred.deinit();

    var faces = try input.faces(allocator);
    defer faces.deinit();

    var carrier_builder = topology.SimplicialCarrierMapBuilder.init(allocator, input, output);
    errdefer carrier_builder.deinit();

    for (faces.simplexes) |face| {
        var image_builder = topology.ComplexBuilder.init(allocator);
        defer image_builder.deinit();

        if (face.isEmpty()) {
            try image_builder.add(.{ .vertices = &[_]topology.Vertex{} });
        } else if (face.isSubsetOf(regular.simplex())) {
            try image_builder.add(committed.simplex());
            try image_builder.add(deferred.simplex());
        } else if (face.isSubsetOf(invalid.simplex())) {
            try image_builder.add(deferred.simplex());
        } else {
            return error.UncoveredAbstractWorkflowFace;
        }

        var image = try image_builder.toOwnedComplex();
        defer image.deinit();
        try carrier_builder.add(face, .{
            .parent = output,
            .simplexes = image.simplexes,
        });
    }

    var carrier = try carrier_builder.toOwned();
    errdefer carrier.deinit();
    try carrier.carrierMap().validate(allocator, .{ .require_monotone = true });
    return carrier;
}

fn validateCarrierRefines(
    allocator: std.mem.Allocator,
    detailed: topology.SimplicialCarrierMap,
    abstract_task: topology.SimplicialCarrierMap,
) !void {
    try detailed.validate(allocator, .{});
    try abstract_task.validate(allocator, .{});
    if (!(try complexesEqualByClosure(allocator, detailed.domain, abstract_task.domain))) {
        return error.RefinementDomainMismatch;
    }
    if (!(try complexesEqualByClosure(allocator, detailed.codomain, abstract_task.codomain))) {
        return error.RefinementCodomainMismatch;
    }

    var faces = try detailed.domain.faces(allocator);
    defer faces.deinit();
    for (faces.simplexes) |face| {
        const detailed_image = detailed.imageOf(face) orelse return error.MissingCarrierSimplex;
        const abstract_image = abstract_task.imageOf(face) orelse return error.MissingCarrierSimplex;
        if (!(try subcomplexSubsetOf(allocator, detailed_image, abstract_image))) {
            return error.DetailedCarrierEscapesAbstractTask;
        }
    }
}

fn subcomplexSubsetOf(
    allocator: std.mem.Allocator,
    left: topology.Subcomplex,
    right: topology.Subcomplex,
) !bool {
    var left_faces = try left.complex().faces(allocator);
    defer left_faces.deinit();

    for (left_faces.simplexes) |simplex| {
        if (!right.complex().containsSimplex(simplex)) return false;
    }
    return true;
}

fn complexesEqualByClosure(
    allocator: std.mem.Allocator,
    left: topology.Complex,
    right: topology.Complex,
) !bool {
    var left_faces = try left.faces(allocator);
    defer left_faces.deinit();
    var right_faces = try right.faces(allocator);
    defer right_faces.deinit();

    for (left_faces.simplexes) |simplex| {
        if (!right_faces.complex().containsSimplex(simplex)) return false;
    }
    for (right_faces.simplexes) |simplex| {
        if (!left_faces.complex().containsSimplex(simplex)) return false;
    }
    return true;
}

fn findVertexByRef(
    allocator: std.mem.Allocator,
    complex: topology.Complex,
    ref: VertexRef,
) !topology.Vertex {
    var vertices = try complex.vertices(allocator);
    defer vertices.deinit();

    for (vertices.vertices) |vertex| {
        if (topology.Color.eql(vertex.color, ref.color) and
            topology.Label.eql(vertex.label, .{ .symbol = ref.label }))
        {
            return vertex;
        }
    }
    return error.VertexRefNotFound;
}

fn simplexFromRefs(
    allocator: std.mem.Allocator,
    complex: topology.Complex,
    refs: []const VertexRef,
) !topology.OwnedSimplex {
    const vertices = try allocator.alloc(topology.Vertex, refs.len);
    errdefer allocator.free(vertices);
    for (refs, 0..) |ref, index| {
        vertices[index] = try findVertexByRef(allocator, complex, ref);
    }
    std.mem.sort(topology.Vertex, vertices, {}, topology.Vertex.lessThan);
    return .{ .allocator = allocator, .vertices = vertices };
}

fn carrierCarriesRefs(
    allocator: std.mem.Allocator,
    carrier: topology.SimplicialCarrierMap,
    input_refs: []const VertexRef,
    output_refs: []const VertexRef,
) !bool {
    var input = try simplexFromRefs(allocator, carrier.domain, input_refs);
    defer input.deinit();
    var output = try simplexFromRefs(allocator, carrier.codomain, output_refs);
    defer output.deinit();
    return carrier.carries(input.simplex(), output.simplex());
}

fn vr(color: topology.Color, label: []const u8) VertexRef {
    return .{ .color = color, .label = label };
}

fn p(
    source_color: topology.Color,
    source_label: []const u8,
    target_color: topology.Color,
    target_label: []const u8,
) ProjectionSpec {
    return .{
        .source = vr(source_color, source_label),
        .target = vr(target_color, target_label),
    };
}

test "projected carrier composition validates workload to RDA chain" {
    var composition = try buildGlobalTrainingComposition(std.testing.allocator);
    defer composition.deinit();

    try composition.composedCarrier().validate(std.testing.allocator, .{ .require_monotone = true });
    try std.testing.expect(try carrierCarriesRefs(
        std.testing.allocator,
        composition.composedCarrier(),
        &.{
            vr(.{ .api_caller = 0 }, "request.regular"),
            vr(.{ .gateway = 0 }, "parser.training.regular"),
            vr(.{ .job = 1 }, "job.training.regular"),
        },
        &.{
            vr(.{ .gateway = 0 }, "rda.output.direction_norm_carried"),
            vr(.{ .session = 50 }, "merge.committed"),
            vr(.{ .worker = 0 }, "participants.used"),
        },
    ));
    try std.testing.expect(try carrierCarriesRefs(
        std.testing.allocator,
        composition.composedCarrier(),
        &.{
            vr(.{ .api_caller = 0 }, "request.regular"),
            vr(.{ .gateway = 0 }, "parser.training.regular"),
            vr(.{ .job = 1 }, "job.training.regular"),
        },
        &.{
            vr(.{ .gateway = 0 }, "rda.output.none"),
            vr(.{ .session = 52 }, "merge.deferred"),
            vr(.{ .worker = 0 }, "participants.insufficient"),
        },
    ));
}

test "invalid workload cannot refine to committed RDA output" {
    var composition = try buildGlobalTrainingComposition(std.testing.allocator);
    defer composition.deinit();

    try std.testing.expect(!try carrierCarriesRefs(
        std.testing.allocator,
        composition.composedCarrier(),
        &.{
            vr(.{ .api_caller = 0 }, "request.invalid"),
            vr(.{ .gateway = 0 }, "parser.training.invalid"),
            vr(.{ .job = 1 }, "job.training.invalid"),
        },
        &.{
            vr(.{ .gateway = 0 }, "rda.output.direction_norm_carried"),
            vr(.{ .session = 50 }, "merge.committed"),
            vr(.{ .worker = 0 }, "participants.used"),
        },
    ));
    try std.testing.expect(try carrierCarriesRefs(
        std.testing.allocator,
        composition.composedCarrier(),
        &.{
            vr(.{ .api_caller = 0 }, "request.invalid"),
            vr(.{ .gateway = 0 }, "parser.training.invalid"),
            vr(.{ .job = 1 }, "job.training.invalid"),
        },
        &.{
            vr(.{ .gateway = 0 }, "rda.output.none"),
            vr(.{ .session = 52 }, "merge.deferred"),
            vr(.{ .worker = 0 }, "participants.insufficient"),
        },
    ));
}

test "RDA to global projection refines detailed pipeline to abstract global carrier" {
    var composition = try buildGlobalTrainingComposition(std.testing.allocator);
    defer composition.deinit();

    try validateProjectedRefinement(
        std.testing.allocator,
        composition.composedCarrier(),
        composition.rda_to_global.vertexMap(),
        composition.abstractCarrier(),
    );
}

test "reservation cleanup composes through fragment rejection into deferred merge" {
    var composition = try buildGlobalTrainingComposition(std.testing.allocator);
    defer composition.deinit();

    const reservation_task = composition.reservation.task();
    const fragment_task = composition.fragment.task();
    const rda_task = composition.rda.task();
    const links = [_]ProjectionCarrierLink{
        .{
            .name = "reservation -> fragment",
            .projection = composition.reservation_to_fragment.vertexMap(),
            .next = fragment_task.carrier,
        },
        .{
            .name = "fragment -> RDA",
            .projection = composition.fragment_to_rda.vertexMap(),
            .next = rda_task.carrier,
        },
    };
    var cleanup = try composeProjectedCarriers(std.testing.allocator, reservation_task.carrier, &links);
    defer cleanup.deinit();

    try std.testing.expect(try carrierCarriesRefs(
        std.testing.allocator,
        cleanup.carrierMap(),
        &.{
            vr(.{ .gateway = 0 }, "queue.reserved_job"),
            vr(.{ .job = 23 }, "reservation.cancel"),
            vr(.{ .worker = 0 }, "worker.leased.queued_job"),
        },
        &.{
            vr(.{ .gateway = 0 }, "rda.output.none"),
            vr(.{ .session = 52 }, "merge.deferred"),
            vr(.{ .worker = 0 }, "participants.insufficient"),
        },
    ));
}
