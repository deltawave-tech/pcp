const std = @import("std");

const topology = @import("topology.zig");

pub const SubdivisionKind = enum {
    identity,
    barycentric,
    standard_chromatic,
};

pub const SubdivisionVertexMetadata = struct {
    vertex: topology.Vertex,
    kind: SubdivisionKind,
    round: usize,
    ordinal: usize,
    source_vertex: ?topology.Vertex = null,
    carrier_vertices: []topology.Vertex,

    pub fn carrierSimplex(self: SubdivisionVertexMetadata) topology.Simplex {
        return .{ .vertices = self.carrier_vertices };
    }
};

pub const OwnedSubdivision = struct {
    allocator: std.mem.Allocator,
    complex: topology.OwnedComplex,
    carrier: topology.OwnedSimplicialCarrierMap,
    metadata: []SubdivisionVertexMetadata,

    pub fn subdividedComplex(self: *const OwnedSubdivision) topology.Complex {
        return self.complex.complex();
    }

    pub fn subdivisionCarrier(self: *const OwnedSubdivision) topology.SimplicialCarrierMap {
        return self.carrier.carrierMap();
    }

    pub fn metadataForVertex(self: *const OwnedSubdivision, vertex: topology.Vertex) ?SubdivisionVertexMetadata {
        for (self.metadata) |metadata| {
            if (topology.Vertex.eql(metadata.vertex, vertex)) return metadata;
        }
        return null;
    }

    pub fn deinit(self: *OwnedSubdivision) void {
        for (self.metadata) |metadata| {
            self.allocator.free(metadata.carrier_vertices);
        }
        self.allocator.free(self.metadata);
        self.carrier.deinit();
        self.complex.deinit();
        self.metadata = &.{};
    }
};

pub const SubdivisionValidationOptions = struct {
    expected_kind: ?SubdivisionKind = null,
    expected_round: ?usize = null,
    require_strict_carrier: bool = true,
    require_rigid_carrier: bool = true,
    require_chromatic_carrier: bool = false,
    validate_boundary: bool = true,
    validate_links: bool = true,
};

pub const SubdivisionValidationReport = struct {
    vertex_count: usize,
    metadata_count: usize,
    top_facet_count: usize,
    face_restriction_checks: usize,
    boundary_checks: usize,
    link_checks: usize,
};

pub fn barycentricSubdivision(
    allocator: std.mem.Allocator,
    input: topology.Complex,
) !OwnedSubdivision {
    return barycentricSubdivisionRound(allocator, input, 1);
}

pub fn standardChromaticSubdivision(
    allocator: std.mem.Allocator,
    input: topology.Complex,
) !OwnedSubdivision {
    return standardChromaticSubdivisionRound(allocator, input, 1);
}

pub fn iteratedSubdivision(
    allocator: std.mem.Allocator,
    input: topology.Complex,
    kind: SubdivisionKind,
    rounds: usize,
) !OwnedSubdivision {
    if (rounds == 0) return identitySubdivision(allocator, input);
    if (kind == .identity) return identitySubdivision(allocator, input);

    var current = try subdivisionRound(allocator, input, kind, 1);
    errdefer current.deinit();

    var round: usize = 2;
    while (round <= rounds) : (round += 1) {
        var next = try subdivisionRound(allocator, current.subdividedComplex(), kind, round);
        errdefer next.deinit();

        var composed = try current.subdivisionCarrier().composeWithCarrier(
            allocator,
            next.subdivisionCarrier(),
        );
        errdefer composed.deinit();

        next.carrier.deinit();
        next.carrier = composed;
        current.deinit();
        current = next;
    }

    return current;
}

pub fn validateSubdivisionProtocolPower(
    allocator: std.mem.Allocator,
    input: topology.Complex,
    subdivision: *const OwnedSubdivision,
    options: SubdivisionValidationOptions,
) !SubdivisionValidationReport {
    try input.validate();
    try subdivision.subdividedComplex().validate();

    try subdivision.subdivisionCarrier().validate(allocator, .{
        .require_strict = options.require_strict_carrier,
        .require_rigid = options.require_rigid_carrier,
        .require_chromatic = options.require_chromatic_carrier,
    });

    try validateSubdivisionMetadata(
        allocator,
        input,
        subdivision.subdividedComplex(),
        subdivision.metadata,
        options,
    );
    try validateSubdivisionSimplexesByMetadata(
        allocator,
        subdivision.subdividedComplex(),
        subdivision.metadata,
        options.expected_kind,
    );
    try validateCarrierImagesAgainstMetadata(
        allocator,
        input,
        subdivision.subdividedComplex(),
        subdivision.subdivisionCarrier(),
        subdivision.metadata,
    );

    var report = SubdivisionValidationReport{
        .vertex_count = 0,
        .metadata_count = subdivision.metadata.len,
        .top_facet_count = try countFacesOfDimension(
            subdivision.subdividedComplex(),
            allocator,
            subdivision.subdividedComplex().dimension(),
        ),
        .face_restriction_checks = try validateCarrierFaceRestrictions(
            allocator,
            input,
            subdivision.subdivisionCarrier(),
            subdivision.metadata,
        ),
        .boundary_checks = 0,
        .link_checks = 0,
    };

    var vertices = try subdivision.subdividedComplex().vertices(allocator);
    defer vertices.deinit();
    report.vertex_count = vertices.vertices.len;

    if (options.validate_boundary) {
        report.boundary_checks = try validateBoundaryCompatibility(
            allocator,
            input,
            subdivision.subdivisionCarrier(),
        );
    }
    if (options.validate_links) {
        report.link_checks = try validateLinkCompatibility(
            allocator,
            subdivision.subdivisionCarrier(),
            subdivision.metadata,
        );
    }

    return report;
}

pub fn validateIteratedSubdivisionEquivalence(
    allocator: std.mem.Allocator,
    input: topology.Complex,
    kind: SubdivisionKind,
    rounds: usize,
) !void {
    if (rounds == 0) return;
    if (kind == .identity) return;

    var iterated = try iteratedSubdivision(allocator, input, kind, rounds);
    defer iterated.deinit();
    try iterated.subdivisionCarrier().validate(allocator, .{ .require_rigid = true });

    var prefix = try iteratedSubdivision(allocator, input, kind, rounds - 1);
    defer prefix.deinit();
    var final_round = try subdivisionRound(allocator, prefix.subdividedComplex(), kind, rounds);
    defer final_round.deinit();

    var composed = try prefix.subdivisionCarrier().composeWithCarrier(
        allocator,
        final_round.subdivisionCarrier(),
    );
    defer composed.deinit();

    if (!(try complexesEqualByClosure(allocator, iterated.subdividedComplex(), final_round.subdividedComplex()))) {
        return error.IteratedSubdivisionComplexMismatch;
    }
    try carriersEqualByClosure(allocator, iterated.subdivisionCarrier(), composed.carrierMap());

    try validateSubdivisionMetadata(
        allocator,
        prefix.subdividedComplex(),
        final_round.subdividedComplex(),
        iterated.metadata,
        .{
            .expected_kind = kind,
            .expected_round = rounds,
            .require_strict_carrier = false,
            .require_rigid_carrier = false,
        },
    );
    try validateSubdivisionSimplexesByMetadata(
        allocator,
        iterated.subdividedComplex(),
        iterated.metadata,
        kind,
    );
}

fn subdivisionRound(
    allocator: std.mem.Allocator,
    input: topology.Complex,
    kind: SubdivisionKind,
    round: usize,
) !OwnedSubdivision {
    return switch (kind) {
        .identity => identitySubdivision(allocator, input),
        .barycentric => barycentricSubdivisionRound(allocator, input, round),
        .standard_chromatic => standardChromaticSubdivisionRound(allocator, input, round),
    };
}

fn identitySubdivision(allocator: std.mem.Allocator, input: topology.Complex) !OwnedSubdivision {
    try input.validate();

    var complex_builder = topology.ComplexBuilder.init(allocator);
    errdefer complex_builder.deinit();
    for (input.simplexes) |simplex| try complex_builder.add(simplex);

    var complex = try complex_builder.toOwnedComplex();
    errdefer complex.deinit();

    var input_faces = try input.faces(allocator);
    defer input_faces.deinit();

    var carrier_builder = topology.SimplicialCarrierMapBuilder.init(allocator, input, complex.complex());
    errdefer carrier_builder.deinit();
    for (input_faces.simplexes) |face| {
        const image_simplexes = [_]topology.Simplex{face};
        try carrier_builder.add(face, .{
            .parent = complex.complex(),
            .simplexes = &image_simplexes,
        });
    }

    var carrier = try carrier_builder.toOwned();
    errdefer carrier.deinit();

    var vertices = try complex.complex().vertices(allocator);
    defer vertices.deinit();
    var metadata = std.ArrayList(SubdivisionVertexMetadata).init(allocator);
    errdefer deinitMetadataList(allocator, &metadata);
    for (vertices.vertices, 0..) |vertex, ordinal| {
        try metadata.append(try makeMetadata(
            allocator,
            vertex,
            .identity,
            0,
            ordinal,
            vertex,
            .{ .vertices = &[_]topology.Vertex{vertex} },
        ));
    }

    return .{
        .allocator = allocator,
        .complex = complex,
        .carrier = carrier,
        .metadata = try metadata.toOwnedSlice(),
    };
}

fn barycentricSubdivisionRound(
    allocator: std.mem.Allocator,
    input: topology.Complex,
    round: usize,
) !OwnedSubdivision {
    try input.validate();

    var input_faces = try input.faces(allocator);
    defer input_faces.deinit();

    var metadata = std.ArrayList(SubdivisionVertexMetadata).init(allocator);
    errdefer deinitMetadataList(allocator, &metadata);

    for (input_faces.simplexes) |face| {
        if (face.isEmpty()) continue;
        if (face.vertices.len - 1 > std.math.maxInt(u16)) return error.SubdivisionTooLarge;

        const vertex = topology.Vertex{
            .color = .{ .process = @intCast(face.vertices.len - 1) },
            .label = .{ .protocol = @intCast(metadata.items.len) },
        };
        try metadata.append(try makeMetadata(
            allocator,
            vertex,
            .barycentric,
            round,
            metadata.items.len,
            null,
            face,
        ));
    }

    var complex_builder = topology.ComplexBuilder.init(allocator);
    errdefer complex_builder.deinit();

    var chain = std.ArrayList(usize).init(allocator);
    defer chain.deinit();
    for (input_faces.simplexes, 0..) |face, face_index| {
        if (face.isEmpty()) continue;
        try addBarycentricChains(input_faces.simplexes, metadata.items, &chain, face_index, &complex_builder);
    }

    var complex = try complex_builder.toOwnedComplex();
    errdefer complex.deinit();

    var carrier = try buildSubdivisionCarrier(allocator, input, complex.complex(), metadata.items);
    errdefer carrier.deinit();

    return .{
        .allocator = allocator,
        .complex = complex,
        .carrier = carrier,
        .metadata = try metadata.toOwnedSlice(),
    };
}

fn standardChromaticSubdivisionRound(
    allocator: std.mem.Allocator,
    input: topology.Complex,
    round: usize,
) !OwnedSubdivision {
    try input.validate();

    var input_faces = try input.faces(allocator);
    defer input_faces.deinit();

    var metadata = std.ArrayList(SubdivisionVertexMetadata).init(allocator);
    errdefer deinitMetadataList(allocator, &metadata);

    for (input_faces.simplexes) |view| {
        if (view.isEmpty()) continue;
        for (view.vertices) |source_vertex| {
            const vertex = topology.Vertex{
                .color = source_vertex.color,
                .label = .{ .protocol = @intCast(metadata.items.len) },
            };
            try metadata.append(try makeMetadata(
                allocator,
                vertex,
                .standard_chromatic,
                round,
                metadata.items.len,
                source_vertex,
                view,
            ));
        }
    }

    var complex_builder = topology.ComplexBuilder.init(allocator);
    errdefer complex_builder.deinit();

    var selected = std.ArrayList(usize).init(allocator);
    defer selected.deinit();
    try addChromaticMetadataSubsets(
        metadata.items,
        @intCast(input.dimension() + 1),
        0,
        &selected,
        &complex_builder,
    );

    var complex = try complex_builder.toOwnedComplex();
    errdefer complex.deinit();

    var carrier = try buildSubdivisionCarrier(allocator, input, complex.complex(), metadata.items);
    errdefer carrier.deinit();

    return .{
        .allocator = allocator,
        .complex = complex,
        .carrier = carrier,
        .metadata = try metadata.toOwnedSlice(),
    };
}

fn addChromaticMetadataSubsets(
    metadata: []const SubdivisionVertexMetadata,
    max_vertices: usize,
    start_index: usize,
    selected: *std.ArrayList(usize),
    complex_builder: *topology.ComplexBuilder,
) !void {
    if (selected.items.len != 0) {
        const vertices = try complex_builder.allocator.alloc(topology.Vertex, selected.items.len);
        defer complex_builder.allocator.free(vertices);
        for (selected.items, 0..) |metadata_index, out_index| {
            vertices[out_index] = metadata[metadata_index].vertex;
        }
        try complex_builder.add(.{ .vertices = vertices });
    }

    if (selected.items.len == max_vertices) return;

    var index = start_index;
    while (index < metadata.len) : (index += 1) {
        if (!canAppendChromaticMetadata(metadata, selected.items, index)) continue;
        try selected.append(index);
        defer _ = selected.pop();
        try addChromaticMetadataSubsets(metadata, max_vertices, index + 1, selected, complex_builder);
    }
}

fn canAppendChromaticMetadata(
    metadata: []const SubdivisionVertexMetadata,
    selected: []const usize,
    candidate_index: usize,
) bool {
    const candidate = metadata[candidate_index];
    for (selected) |selected_index| {
        const existing = metadata[selected_index];
        if (topology.Color.eql(existing.vertex.color, candidate.vertex.color)) return false;
        if (!viewsComparable(existing.carrierSimplex(), candidate.carrierSimplex())) return false;
        if (!immediateSnapshotRule(existing, candidate)) return false;
        if (!immediateSnapshotRule(candidate, existing)) return false;
    }
    return true;
}

fn addBarycentricChains(
    original_faces: []const topology.Simplex,
    metadata: []const SubdivisionVertexMetadata,
    chain: *std.ArrayList(usize),
    current_face_index: usize,
    complex_builder: *topology.ComplexBuilder,
) !void {
    try chain.append(current_face_index);
    defer _ = chain.pop();

    const chain_vertices = try complex_builder.allocator.alloc(topology.Vertex, chain.items.len);
    defer complex_builder.allocator.free(chain_vertices);
    for (chain.items, 0..) |face_index, out_index| {
        const metadata_index = metadataIndexForCarrier(metadata, original_faces[face_index]) orelse {
            return error.MissingSubdivisionMetadata;
        };
        chain_vertices[out_index] = metadata[metadata_index].vertex;
    }
    try complex_builder.add(.{ .vertices = chain_vertices });

    const current = original_faces[current_face_index];
    for (original_faces, 0..) |candidate, candidate_index| {
        if (candidate.isEmpty()) continue;
        if (!current.isProperSubsetOf(candidate)) continue;
        try addBarycentricChains(original_faces, metadata, chain, candidate_index, complex_builder);
    }
}

fn buildSubdivisionCarrier(
    allocator: std.mem.Allocator,
    input: topology.Complex,
    subdivision: topology.Complex,
    metadata: []const SubdivisionVertexMetadata,
) !topology.OwnedSimplicialCarrierMap {
    var input_faces = try input.faces(allocator);
    defer input_faces.deinit();
    var subdivision_faces = try subdivision.faces(allocator);
    defer subdivision_faces.deinit();

    var carrier_builder = topology.SimplicialCarrierMapBuilder.init(allocator, input, subdivision);
    errdefer carrier_builder.deinit();

    for (input_faces.simplexes) |input_face| {
        var image_builder = topology.ComplexBuilder.init(allocator);
        defer image_builder.deinit();

        for (subdivision_faces.simplexes) |subdivision_face| {
            if (subdivisionFaceCarriedBy(input_face, subdivision_face, metadata)) {
                try image_builder.add(subdivision_face);
            }
        }

        var image = try image_builder.toOwnedComplex();
        defer image.deinit();
        try carrier_builder.add(input_face, .{
            .parent = subdivision,
            .simplexes = image.simplexes,
        });
    }

    return carrier_builder.toOwned();
}

fn validateSubdivisionMetadata(
    allocator: std.mem.Allocator,
    carrier_domain: topology.Complex,
    subdivision: topology.Complex,
    metadata: []const SubdivisionVertexMetadata,
    options: SubdivisionValidationOptions,
) !void {
    var vertices = try subdivision.vertices(allocator);
    defer vertices.deinit();
    if (vertices.vertices.len != metadata.len) return error.SubdivisionMetadataCountMismatch;

    const seen_ordinals = try allocator.alloc(bool, metadata.len);
    defer allocator.free(seen_ordinals);
    @memset(seen_ordinals, false);

    for (metadata) |item| {
        if (!vertexSetContains(vertices.vertices, item.vertex)) return error.SubdivisionMetadataVertexOutsideComplex;
        if (item.ordinal >= metadata.len) return error.SubdivisionMetadataOrdinalOutOfRange;
        if (seen_ordinals[item.ordinal]) return error.DuplicateSubdivisionMetadataOrdinal;
        seen_ordinals[item.ordinal] = true;

        if (options.expected_kind) |kind| {
            if (item.kind != kind) return error.SubdivisionMetadataKindMismatch;
        }
        if (options.expected_round) |round| {
            if (item.round != round) return error.SubdivisionMetadataRoundMismatch;
        }

        const carrier = item.carrierSimplex();
        try carrier.validate();
        if (carrier.isEmpty()) return error.EmptySubdivisionCarrierFace;
        if (!carrier_domain.containsSimplex(carrier)) return error.SubdivisionMetadataCarrierOutsideInput;

        switch (item.kind) {
            .identity => {
                const source = item.source_vertex orelse return error.MissingSubdivisionSourceVertex;
                if (!topology.Vertex.eql(item.vertex, source)) return error.InvalidIdentitySubdivisionMetadata;
                if (carrier.vertices.len != 1 or !carrier.containsVertex(source)) {
                    return error.InvalidIdentitySubdivisionMetadata;
                }
            },
            .barycentric => {
                if (item.source_vertex != null) return error.InvalidBarycentricSubdivisionMetadata;
                switch (item.vertex.color) {
                    .process => |dimension| {
                        if (dimension != carrier.vertices.len - 1) return error.InvalidBarycentricSubdivisionMetadata;
                    },
                    else => return error.InvalidBarycentricSubdivisionMetadata,
                }
            },
            .standard_chromatic => {
                const source = item.source_vertex orelse return error.MissingSubdivisionSourceVertex;
                if (!carrier.containsVertex(source)) return error.SubdivisionSourceOutsideCarrier;
                if (!topology.Color.eql(item.vertex.color, source.color)) return error.NonChromaticSubdivisionMetadata;
            },
        }
    }

    for (vertices.vertices) |vertex| {
        if (metadataForVertex(metadata, vertex) == null) return error.MissingSubdivisionMetadata;
    }
}

fn validateSubdivisionSimplexesByMetadata(
    allocator: std.mem.Allocator,
    subdivision: topology.Complex,
    metadata: []const SubdivisionVertexMetadata,
    expected_kind: ?SubdivisionKind,
) !void {
    var faces = try subdivision.faces(allocator);
    defer faces.deinit();

    for (faces.simplexes) |face| {
        if (face.isEmpty()) continue;
        const first = metadataForVertex(metadata, face.vertices[0]) orelse return error.MissingSubdivisionMetadata;
        const kind = expected_kind orelse first.kind;
        for (face.vertices) |vertex| {
            const item = metadataForVertex(metadata, vertex) orelse return error.MissingSubdivisionMetadata;
            if (item.kind != kind) return error.MixedSubdivisionKindSimplex;
        }

        switch (kind) {
            .identity => try validateIdentityFaceByMetadata(face, metadata),
            .barycentric => try validateBarycentricFaceByMetadata(face, metadata),
            .standard_chromatic => try validateChromaticFaceByMetadata(face, metadata),
        }
    }
}

fn validateIdentityFaceByMetadata(
    face: topology.Simplex,
    metadata: []const SubdivisionVertexMetadata,
) !void {
    for (face.vertices) |vertex| {
        const item = metadataForVertex(metadata, vertex) orelse return error.MissingSubdivisionMetadata;
        const source = item.source_vertex orelse return error.MissingSubdivisionSourceVertex;
        if (!topology.Vertex.eql(vertex, source)) return error.InvalidIdentitySubdivisionMetadata;
    }
}

fn validateBarycentricFaceByMetadata(
    face: topology.Simplex,
    metadata: []const SubdivisionVertexMetadata,
) !void {
    for (face.vertices, 0..) |left_vertex, left_index| {
        const left = metadataForVertex(metadata, left_vertex) orelse return error.MissingSubdivisionMetadata;
        for (face.vertices[left_index + 1 ..]) |right_vertex| {
            const right = metadataForVertex(metadata, right_vertex) orelse return error.MissingSubdivisionMetadata;
            if (topology.Simplex.eql(left.carrierSimplex(), right.carrierSimplex())) {
                return error.InvalidBarycentricSubdivisionChain;
            }
            if (!viewsComparable(left.carrierSimplex(), right.carrierSimplex())) {
                return error.InvalidBarycentricSubdivisionChain;
            }
        }
    }
}

fn validateChromaticFaceByMetadata(
    face: topology.Simplex,
    metadata: []const SubdivisionVertexMetadata,
) !void {
    for (face.vertices, 0..) |left_vertex, left_index| {
        const left = metadataForVertex(metadata, left_vertex) orelse return error.MissingSubdivisionMetadata;
        _ = left.source_vertex orelse return error.MissingSubdivisionSourceVertex;
        for (face.vertices[left_index + 1 ..]) |right_vertex| {
            const right = metadataForVertex(metadata, right_vertex) orelse return error.MissingSubdivisionMetadata;
            _ = right.source_vertex orelse return error.MissingSubdivisionSourceVertex;
            if (!viewsComparable(left.carrierSimplex(), right.carrierSimplex())) {
                return error.InvalidChromaticSubdivisionViewOrder;
            }
            if (!immediateSnapshotRule(left, right)) return error.InvalidChromaticSubdivisionSnapshotRule;
            if (!immediateSnapshotRule(right, left)) return error.InvalidChromaticSubdivisionSnapshotRule;
        }
    }
}

fn validateCarrierImagesAgainstMetadata(
    allocator: std.mem.Allocator,
    input: topology.Complex,
    subdivision: topology.Complex,
    carrier: topology.SimplicialCarrierMap,
    metadata: []const SubdivisionVertexMetadata,
) !void {
    var input_faces = try input.faces(allocator);
    defer input_faces.deinit();
    var subdivision_faces = try subdivision.faces(allocator);
    defer subdivision_faces.deinit();

    for (input_faces.simplexes) |input_face| {
        const actual = carrier.imageOf(input_face) orelse return error.MissingCarrierSimplex;

        var builder = topology.ComplexBuilder.init(allocator);
        defer builder.deinit();
        for (subdivision_faces.simplexes) |subdivision_face| {
            if (metadataFaceCarriedByInputFace(input_face, subdivision_face, metadata)) {
                try builder.add(subdivision_face);
            }
        }

        var expected = try builder.toOwnedComplex();
        defer expected.deinit();
        if (!(try subcomplexEqualsComplexByClosure(allocator, actual, expected.complex()))) {
            return error.SubdivisionCarrierImageMismatch;
        }
    }
}

fn validateCarrierFaceRestrictions(
    allocator: std.mem.Allocator,
    input: topology.Complex,
    carrier: topology.SimplicialCarrierMap,
    metadata: []const SubdivisionVertexMetadata,
) !usize {
    var input_faces = try input.faces(allocator);
    defer input_faces.deinit();

    var checks: usize = 0;
    for (input_faces.simplexes) |tau| {
        const tau_image = carrier.imageOf(tau) orelse return error.MissingCarrierSimplex;
        for (input_faces.simplexes) |sigma| {
            if (!tau.isSubsetOf(sigma)) continue;
            checks += 1;

            const sigma_image = carrier.imageOf(sigma) orelse return error.MissingCarrierSimplex;
            if (!(try subcomplexSubsetOf(allocator, tau_image, sigma_image))) {
                return error.SubdivisionFaceCarrierNotMonotone;
            }

            var restricted = try restrictSubcomplexToCarrierFace(allocator, sigma_image, metadata, tau);
            defer restricted.deinit();
            if (!(try subcomplexesEqualByClosure(allocator, tau_image, restricted.subcomplex()))) {
                return error.SubdivisionFaceRestrictionMismatch;
            }
        }
    }
    return checks;
}

fn validateBoundaryCompatibility(
    allocator: std.mem.Allocator,
    input: topology.Complex,
    carrier: topology.SimplicialCarrierMap,
) !usize {
    var input_faces = try input.faces(allocator);
    defer input_faces.deinit();

    var checks: usize = 0;
    for (input_faces.simplexes) |sigma| {
        if (sigma.dimension() <= 0) continue;
        checks += 1;

        const sigma_image = carrier.imageOf(sigma) orelse return error.MissingCarrierSimplex;
        var boundary = try sigma_image.complex().boundary(allocator);
        defer boundary.deinit();

        var union_builder = topology.ComplexBuilder.init(allocator);
        defer union_builder.deinit();
        for (input_faces.simplexes) |facet| {
            if (!facet.isSubsetOf(sigma)) continue;
            if (facet.vertices.len + 1 != sigma.vertices.len) continue;
            const facet_image = carrier.imageOf(facet) orelse return error.MissingCarrierSimplex;
            var facet_faces = try facet_image.complex().faces(allocator);
            defer facet_faces.deinit();
            for (facet_faces.simplexes) |face| try union_builder.add(face);
        }

        var expected = try union_builder.toOwnedComplex();
        defer expected.deinit();
        if (!(try complexesEqualByClosure(allocator, boundary.complex(), expected.complex()))) {
            return error.SubdivisionBoundaryMismatch;
        }
    }
    return checks;
}

fn validateLinkCompatibility(
    allocator: std.mem.Allocator,
    carrier: topology.SimplicialCarrierMap,
    metadata: []const SubdivisionVertexMetadata,
) !usize {
    var checks: usize = 0;
    for (carrier.entries) |entry| {
        for (metadata) |item| {
            const center_vertices = [_]topology.Vertex{item.vertex};
            const center = topology.Simplex{ .vertices = &center_vertices };
            if (!entry.image.containsSimplex(center)) continue;
            checks += 1;

            var link = try entry.image.complex().link(allocator, center);
            defer link.deinit();
            try link.complex().validate();
            for (link.simplexes) |link_face| {
                if (!link_face.disjointFrom(center)) return error.SubdivisionLinkIntersectsCenter;
                var joined = try unionSimplexesLocal(allocator, link_face, center);
                defer joined.deinit();
                if (!entry.image.containsSimplex(joined.simplex())) return error.SubdivisionLinkOutsideCarrierImage;
            }
        }
    }
    return checks;
}

fn restrictSubcomplexToCarrierFace(
    allocator: std.mem.Allocator,
    subcomplex: topology.Subcomplex,
    metadata: []const SubdivisionVertexMetadata,
    carrier_face: topology.Simplex,
) !topology.OwnedSubcomplex {
    var faces = try subcomplex.complex().faces(allocator);
    defer faces.deinit();

    var builder = topology.ComplexBuilder.init(allocator);
    errdefer builder.deinit();
    for (faces.simplexes) |face| {
        if (metadataFaceCarriedByInputFace(carrier_face, face, metadata)) {
            try builder.add(face);
        }
    }

    var owned = try builder.toOwnedComplex();
    errdefer owned.deinit();
    return .{ .parent = subcomplex.parent, .owned = owned };
}

fn metadataFaceCarriedByInputFace(
    input_face: topology.Simplex,
    subdivision_face: topology.Simplex,
    metadata: []const SubdivisionVertexMetadata,
) bool {
    if (input_face.isEmpty()) return subdivision_face.isEmpty();
    if (subdivision_face.isEmpty()) return true;

    for (subdivision_face.vertices) |vertex| {
        const item = metadataForVertex(metadata, vertex) orelse return false;
        if (!item.carrierSimplex().isSubsetOf(input_face)) return false;
    }
    return true;
}

fn subdivisionFaceCarriedBy(
    input_face: topology.Simplex,
    subdivision_face: topology.Simplex,
    metadata: []const SubdivisionVertexMetadata,
) bool {
    if (input_face.isEmpty()) return subdivision_face.isEmpty();
    if (subdivision_face.isEmpty()) return true;

    for (subdivision_face.vertices) |vertex| {
        const item = metadataForVertex(metadata, vertex) orelse return false;
        if (!item.carrierSimplex().isSubsetOf(input_face)) return false;
    }
    return true;
}

fn chromaticSubsetIsValid(metadata: []const SubdivisionVertexMetadata, mask: usize) bool {
    var selected_count: usize = 0;
    for (metadata, 0..) |left, left_index| {
        const left_bit = @as(usize, 1) << @intCast(left_index);
        if ((mask & left_bit) == 0) continue;
        selected_count += 1;

        for (metadata[left_index + 1 ..], left_index + 1..) |right, right_index| {
            const right_bit = @as(usize, 1) << @intCast(right_index);
            if ((mask & right_bit) == 0) continue;

            if (topology.Color.eql(left.vertex.color, right.vertex.color)) return false;
            if (!viewsComparable(left.carrierSimplex(), right.carrierSimplex())) return false;
            if (!immediateSnapshotRule(left, right)) return false;
            if (!immediateSnapshotRule(right, left)) return false;
        }
    }

    return selected_count != 0;
}

fn viewsComparable(left: topology.Simplex, right: topology.Simplex) bool {
    return left.isSubsetOf(right) or right.isSubsetOf(left);
}

fn immediateSnapshotRule(left: SubdivisionVertexMetadata, right: SubdivisionVertexMetadata) bool {
    const left_source = left.source_vertex orelse return false;
    if (!right.carrierSimplex().containsVertex(left_source)) return true;
    return left.carrierSimplex().isSubsetOf(right.carrierSimplex());
}

fn makeMetadata(
    allocator: std.mem.Allocator,
    vertex: topology.Vertex,
    kind: SubdivisionKind,
    round: usize,
    ordinal: usize,
    source_vertex: ?topology.Vertex,
    carrier: topology.Simplex,
) !SubdivisionVertexMetadata {
    var canonical = try carrier.canonicalize(allocator);
    errdefer canonical.deinit();
    const carrier_vertices = canonical.vertices;
    canonical.vertices = &.{};

    return .{
        .vertex = vertex,
        .kind = kind,
        .round = round,
        .ordinal = ordinal,
        .source_vertex = source_vertex,
        .carrier_vertices = carrier_vertices,
    };
}

fn metadataIndexForCarrier(
    metadata: []const SubdivisionVertexMetadata,
    carrier: topology.Simplex,
) ?usize {
    for (metadata, 0..) |item, index| {
        if (topology.Simplex.eql(item.carrierSimplex(), carrier)) return index;
    }
    return null;
}

fn metadataForVertex(
    metadata: []const SubdivisionVertexMetadata,
    vertex: topology.Vertex,
) ?SubdivisionVertexMetadata {
    for (metadata) |item| {
        if (topology.Vertex.eql(item.vertex, vertex)) return item;
    }
    return null;
}

fn deinitMetadataList(
    allocator: std.mem.Allocator,
    metadata: *std.ArrayList(SubdivisionVertexMetadata),
) void {
    for (metadata.items) |item| {
        allocator.free(item.carrier_vertices);
    }
    metadata.deinit();
}

fn countFacesOfDimension(complex: topology.Complex, allocator: std.mem.Allocator, dimension: isize) !usize {
    var faces = try complex.faces(allocator);
    defer faces.deinit();

    var count: usize = 0;
    for (faces.simplexes) |simplex| {
        if (simplex.dimension() == dimension) count += 1;
    }
    return count;
}

fn subcomplexSubsetOf(
    allocator: std.mem.Allocator,
    left: topology.Subcomplex,
    right: topology.Subcomplex,
) !bool {
    var left_faces = try left.complex().faces(allocator);
    defer left_faces.deinit();

    for (left_faces.simplexes) |face| {
        if (!right.containsSimplex(face)) return false;
    }
    return true;
}

fn subcomplexesEqualByClosure(
    allocator: std.mem.Allocator,
    left: topology.Subcomplex,
    right: topology.Subcomplex,
) !bool {
    return (try subcomplexSubsetOf(allocator, left, right)) and
        (try subcomplexSubsetOf(allocator, right, left));
}

fn subcomplexEqualsComplexByClosure(
    allocator: std.mem.Allocator,
    left: topology.Subcomplex,
    right: topology.Complex,
) !bool {
    return try subcomplexesEqualByClosure(allocator, left, right.asSubcomplex());
}

fn complexesEqualByClosure(
    allocator: std.mem.Allocator,
    left: topology.Complex,
    right: topology.Complex,
) !bool {
    return try subcomplexesEqualByClosure(allocator, left.asSubcomplex(), right.asSubcomplex());
}

fn carriersEqualByClosure(
    allocator: std.mem.Allocator,
    left: topology.SimplicialCarrierMap,
    right: topology.SimplicialCarrierMap,
) !void {
    if (!(try complexesEqualByClosure(allocator, left.domain, right.domain))) {
        return error.SubdivisionCarrierDomainMismatch;
    }
    if (!(try complexesEqualByClosure(allocator, left.codomain, right.codomain))) {
        return error.SubdivisionCarrierCodomainMismatch;
    }

    var domain_faces = try left.domain.faces(allocator);
    defer domain_faces.deinit();
    for (domain_faces.simplexes) |face| {
        const left_image = left.imageOf(face) orelse return error.MissingCarrierSimplex;
        const right_image = right.imageOf(face) orelse return error.MissingCarrierSimplex;
        if (!(try subcomplexesEqualByClosure(allocator, left_image, right_image))) {
            return error.SubdivisionCarrierImageMismatch;
        }
    }
}

fn vertexSetContains(vertices: []const topology.Vertex, vertex: topology.Vertex) bool {
    for (vertices) |candidate| {
        if (topology.Vertex.eql(candidate, vertex)) return true;
    }
    return false;
}

fn unionSimplexesLocal(
    allocator: std.mem.Allocator,
    left: topology.Simplex,
    right: topology.Simplex,
) !topology.OwnedSimplex {
    var vertices = std.ArrayList(topology.Vertex).init(allocator);
    errdefer vertices.deinit();
    for (left.vertices) |vertex| {
        if (!vertexSetContains(vertices.items, vertex)) try vertices.append(vertex);
    }
    for (right.vertices) |vertex| {
        if (!vertexSetContains(vertices.items, vertex)) try vertices.append(vertex);
    }
    std.mem.sort(topology.Vertex, vertices.items, {}, topology.Vertex.lessThan);
    return .{ .allocator = allocator, .vertices = try vertices.toOwnedSlice() };
}

test "barycentric subdivision of an edge has two subdivided edges and a rigid carrier" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const edge_vertices = [_]topology.Vertex{ a, b };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &edge_vertices }} };

    var subdivision = try barycentricSubdivision(std.testing.allocator, input);
    defer subdivision.deinit();

    var vertices = try subdivision.subdividedComplex().vertices(std.testing.allocator);
    defer vertices.deinit();

    try std.testing.expectEqual(@as(usize, 3), vertices.vertices.len);
    try std.testing.expectEqual(@as(usize, 2), try countFacesOfDimension(subdivision.subdividedComplex(), std.testing.allocator, 1));
    try subdivision.subdivisionCarrier().validate(std.testing.allocator, .{
        .require_strict = true,
        .require_rigid = true,
    });
}

test "standard chromatic subdivision of an edge matches the book picture" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const edge_vertices = [_]topology.Vertex{ a, b };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &edge_vertices }} };

    var subdivision = try standardChromaticSubdivision(std.testing.allocator, input);
    defer subdivision.deinit();

    var vertices = try subdivision.subdividedComplex().vertices(std.testing.allocator);
    defer vertices.deinit();

    try std.testing.expectEqual(@as(usize, 4), vertices.vertices.len);
    try std.testing.expectEqual(@as(usize, 3), try countFacesOfDimension(subdivision.subdividedComplex(), std.testing.allocator, 1));
    try subdivision.subdivisionCarrier().validate(std.testing.allocator, .{
        .require_strict = true,
        .require_rigid = true,
        .require_chromatic = true,
    });
}

test "standard chromatic subdivision of a triangle has twelve vertices and thirteen facets" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = topology.Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const triangle_vertices = [_]topology.Vertex{ a, b, c };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &triangle_vertices }} };

    var subdivision = try standardChromaticSubdivision(std.testing.allocator, input);
    defer subdivision.deinit();

    var vertices = try subdivision.subdividedComplex().vertices(std.testing.allocator);
    defer vertices.deinit();

    try std.testing.expectEqual(@as(usize, 12), vertices.vertices.len);
    try std.testing.expectEqual(@as(usize, 13), try countFacesOfDimension(subdivision.subdividedComplex(), std.testing.allocator, 2));
    try subdivision.subdivisionCarrier().validate(std.testing.allocator, .{
        .require_strict = true,
        .require_rigid = true,
        .require_chromatic = true,
    });
}

test "iterated barycentric subdivision composes carriers across rounds" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const edge_vertices = [_]topology.Vertex{ a, b };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &edge_vertices }} };

    var subdivision = try iteratedSubdivision(std.testing.allocator, input, .barycentric, 2);
    defer subdivision.deinit();

    try std.testing.expectEqual(@as(usize, 4), try countFacesOfDimension(subdivision.subdividedComplex(), std.testing.allocator, 1));
    try subdivision.subdivisionCarrier().validate(std.testing.allocator, .{
        .require_rigid = true,
    });
    for (subdivision.metadata) |metadata| {
        try std.testing.expectEqual(@as(usize, 2), metadata.round);
    }
}

test "subdivision metadata explains chromatic views and carrier faces" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const edge_vertices = [_]topology.Vertex{ a, b };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &edge_vertices }} };

    var subdivision = try standardChromaticSubdivision(std.testing.allocator, input);
    defer subdivision.deinit();

    var found_joint_view = false;
    for (subdivision.metadata) |metadata| {
        try std.testing.expectEqual(SubdivisionKind.standard_chromatic, metadata.kind);
        try std.testing.expectEqual(@as(usize, 1), metadata.round);
        if (metadata.carrier_vertices.len == 2 and metadata.source_vertex != null) {
            found_joint_view = true;
            try std.testing.expect(metadata.carrierSimplex().containsVertex(metadata.source_vertex.?));
        }
    }
    try std.testing.expect(found_joint_view);
}

test "protocol-power validator accepts canonical chromatic edge triangle and tetrahedron subdivisions" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = topology.Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const d = topology.Vertex{ .color = .{ .process = 3 }, .label = .{ .input = 3 } };
    const edge_vertices = [_]topology.Vertex{ a, b };
    const triangle_vertices = [_]topology.Vertex{ a, b, c };
    const tetrahedron_vertices = [_]topology.Vertex{ a, b, c, d };

    const edge = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &edge_vertices }} };
    const triangle = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &triangle_vertices }} };
    const tetrahedron = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &tetrahedron_vertices }} };

    const options = SubdivisionValidationOptions{
        .expected_kind = .standard_chromatic,
        .expected_round = 1,
        .require_chromatic_carrier = true,
    };

    var edge_subdivision = try standardChromaticSubdivision(std.testing.allocator, edge);
    defer edge_subdivision.deinit();
    const edge_report = try validateSubdivisionProtocolPower(std.testing.allocator, edge, &edge_subdivision, options);
    try std.testing.expectEqual(@as(usize, 4), edge_report.vertex_count);
    try std.testing.expectEqual(@as(usize, 3), edge_report.top_facet_count);
    try std.testing.expect(edge_report.face_restriction_checks > 0);
    try std.testing.expect(edge_report.boundary_checks > 0);
    try std.testing.expect(edge_report.link_checks > 0);

    var triangle_subdivision = try standardChromaticSubdivision(std.testing.allocator, triangle);
    defer triangle_subdivision.deinit();
    const triangle_report = try validateSubdivisionProtocolPower(std.testing.allocator, triangle, &triangle_subdivision, options);
    try std.testing.expectEqual(@as(usize, 12), triangle_report.vertex_count);
    try std.testing.expectEqual(@as(usize, 13), triangle_report.top_facet_count);

    var tetrahedron_subdivision = try standardChromaticSubdivision(std.testing.allocator, tetrahedron);
    defer tetrahedron_subdivision.deinit();
    const tetrahedron_report = try validateSubdivisionProtocolPower(std.testing.allocator, tetrahedron, &tetrahedron_subdivision, options);
    try std.testing.expectEqual(@as(usize, 32), tetrahedron_report.vertex_count);
    try std.testing.expectEqual(@as(usize, 75), tetrahedron_report.top_facet_count);
}

test "protocol-power validator checks barycentric tetrahedron boundary and links" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = topology.Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const d = topology.Vertex{ .color = .{ .process = 3 }, .label = .{ .input = 3 } };
    const tetrahedron_vertices = [_]topology.Vertex{ a, b, c, d };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &tetrahedron_vertices }} };

    var subdivision = try barycentricSubdivision(std.testing.allocator, input);
    defer subdivision.deinit();
    const report = try validateSubdivisionProtocolPower(std.testing.allocator, input, &subdivision, .{
        .expected_kind = .barycentric,
        .expected_round = 1,
    });

    try std.testing.expectEqual(@as(usize, 15), report.vertex_count);
    try std.testing.expectEqual(@as(usize, 24), report.top_facet_count);
    try std.testing.expect(report.boundary_checks >= 11);
    try std.testing.expect(report.link_checks > 0);
}

test "protocol-power validator accepts mixed-complex chromatic subdivisions" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = topology.Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const d = topology.Vertex{ .color = .{ .process = 3 }, .label = .{ .input = 3 } };
    const triangle_vertices = [_]topology.Vertex{ a, b, c };
    const edge_vertices = [_]topology.Vertex{ c, d };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &triangle_vertices },
        .{ .vertices = &edge_vertices },
    } };

    var subdivision = try standardChromaticSubdivision(std.testing.allocator, input);
    defer subdivision.deinit();
    const report = try validateSubdivisionProtocolPower(std.testing.allocator, input, &subdivision, .{
        .expected_kind = .standard_chromatic,
        .expected_round = 1,
        .require_chromatic_carrier = true,
    });

    try std.testing.expect(report.vertex_count > 12);
    try std.testing.expect(report.top_facet_count >= 13);
    try std.testing.expect(report.boundary_checks > 0);
}

test "iterated subdivision equivalence checks bounded rounds" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const edge_vertices = [_]topology.Vertex{ a, b };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &edge_vertices }} };

    try validateIteratedSubdivisionEquivalence(std.testing.allocator, input, .barycentric, 2);
    try validateIteratedSubdivisionEquivalence(std.testing.allocator, input, .standard_chromatic, 2);
}
