const std = @import("std");

const topology = @import("topology.zig");

pub const OwnedSkeleton = topology.OwnedComplex;

pub const OwnedAdjacencyGraph = struct {
    allocator: std.mem.Allocator,
    vertices: []topology.Vertex,
    adjacency: []bool,

    pub fn vertexCount(self: OwnedAdjacencyGraph) usize {
        return self.vertices.len;
    }

    pub fn adjacent(self: OwnedAdjacencyGraph, left: usize, right: usize) bool {
        return self.adjacency[left * self.vertices.len + right];
    }

    pub fn deinit(self: *OwnedAdjacencyGraph) void {
        self.allocator.free(self.vertices);
        self.allocator.free(self.adjacency);
        self.vertices = &.{};
        self.adjacency = &.{};
    }
};

pub const OwnedComponents = struct {
    allocator: std.mem.Allocator,
    component_ids: []usize,
    component_count: usize,

    pub fn deinit(self: *OwnedComponents) void {
        self.allocator.free(self.component_ids);
        self.component_ids = &.{};
        self.component_count = 0;
    }
};

pub const OwnedVertexPath = struct {
    allocator: std.mem.Allocator,
    vertices: []topology.Vertex,

    pub fn deinit(self: *OwnedVertexPath) void {
        self.allocator.free(self.vertices);
        self.vertices = &.{};
    }
};

pub const OwnedBettiNumbers = struct {
    allocator: std.mem.Allocator,
    values: []usize,

    pub fn deinit(self: *OwnedBettiNumbers) void {
        self.allocator.free(self.values);
        self.values = &.{};
    }
};

pub const HomologyMapReport = struct {
    allocator: std.mem.Allocator,
    source_betti: []usize,
    image_betti: []usize,
    induced_ranks: []usize,
    preserves_betti: bool,
    preserves_homology: bool,

    pub fn deinit(self: *HomologyMapReport) void {
        self.allocator.free(self.source_betti);
        self.allocator.free(self.image_betti);
        self.allocator.free(self.induced_ranks);
        self.source_betti = &.{};
        self.image_betti = &.{};
        self.induced_ranks = &.{};
    }
};

pub const HomologyMapWitnessKind = enum {
    source_cycle_lost,
    image_cycle_created,
};

pub const OwnedChain = struct {
    allocator: std.mem.Allocator,
    dimension: usize,
    prime: u16,
    simplexes: []topology.Simplex,
    coefficients: []u16,

    pub fn deinit(self: *OwnedChain) void {
        for (self.simplexes) |simplex| {
            self.allocator.free(@constCast(simplex.vertices));
        }
        self.allocator.free(self.simplexes);
        self.allocator.free(self.coefficients);
        self.simplexes = &.{};
        self.coefficients = &.{};
    }
};

pub const OwnedHomologyMapWitness = struct {
    allocator: std.mem.Allocator,
    kind: HomologyMapWitnessKind,
    dimension: usize,
    prime: u16,
    source_cycle: ?OwnedChain = null,
    image_cycle: ?OwnedChain = null,

    pub fn deinit(self: *OwnedHomologyMapWitness) void {
        if (self.source_cycle) |*cycle| cycle.deinit();
        if (self.image_cycle) |*cycle| cycle.deinit();
        self.source_cycle = null;
        self.image_cycle = null;
    }
};

pub const EdgeGenerator = struct {
    name: usize,
    from: topology.Vertex,
    to: topology.Vertex,
};

pub const SignedGenerator = struct {
    generator_index: usize,
    inverse: bool,
};

pub const Relator = struct {
    word: []SignedGenerator,
};

pub const FundamentalGroupPresentation = struct {
    allocator: std.mem.Allocator,
    generators: []EdgeGenerator,
    relators: []Relator,

    pub fn deinit(self: *FundamentalGroupPresentation) void {
        for (self.relators) |relator| {
            self.allocator.free(relator.word);
        }
        self.allocator.free(self.relators);
        self.allocator.free(self.generators);
        self.relators = &.{};
        self.generators = &.{};
    }
};

pub const ConnectivityObstructionKind = enum {
    none,
    output_disconnected_for_connected_protocol,
};

pub const OwnedConnectivityObstructionReport = struct {
    allocator: std.mem.Allocator,
    kind: ConnectivityObstructionKind,
    input_vertices: []topology.Vertex,
    protocol_components: usize,
    output_components: usize,

    pub fn inputSimplex(self: OwnedConnectivityObstructionReport) topology.Simplex {
        return .{ .vertices = self.input_vertices };
    }

    pub fn deinit(self: *OwnedConnectivityObstructionReport) void {
        self.allocator.free(self.input_vertices);
        self.input_vertices = &.{};
        self.kind = .none;
        self.protocol_components = 0;
        self.output_components = 0;
    }
};

pub fn kSkeleton(
    allocator: std.mem.Allocator,
    complex: topology.Complex,
    max_dimension: isize,
) !OwnedSkeleton {
    return complex.skeleton(allocator, max_dimension);
}

pub fn oneSkeletonAdjacency(
    allocator: std.mem.Allocator,
    complex: topology.Complex,
) !OwnedAdjacencyGraph {
    try complex.validate();

    var vertex_set = try complex.vertices(allocator);
    defer vertex_set.deinit();

    const vertex_count = vertex_set.vertices.len;
    const adjacency = try allocator.alloc(bool, vertex_count * vertex_count);
    errdefer allocator.free(adjacency);
    @memset(adjacency, false);

    for (0..vertex_count) |index| {
        adjacency[index * vertex_count + index] = true;
    }

    for (0..vertex_count) |left_index| {
        for (left_index + 1..vertex_count) |right_index| {
            const edge_vertices = [_]topology.Vertex{
                vertex_set.vertices[left_index],
                vertex_set.vertices[right_index],
            };
            if (complex.containsSimplex(.{ .vertices = &edge_vertices })) {
                adjacency[left_index * vertex_count + right_index] = true;
                adjacency[right_index * vertex_count + left_index] = true;
            }
        }
    }

    return .{
        .allocator = allocator,
        .vertices = try allocator.dupe(topology.Vertex, vertex_set.vertices),
        .adjacency = adjacency,
    };
}

pub fn connectedComponents(
    allocator: std.mem.Allocator,
    complex: topology.Complex,
) !OwnedComponents {
    var graph = try oneSkeletonAdjacency(allocator, complex);
    defer graph.deinit();

    const ids = try allocator.alloc(usize, graph.vertexCount());
    errdefer allocator.free(ids);
    @memset(ids, std.math.maxInt(usize));

    var queue = std.ArrayList(usize).init(allocator);
    defer queue.deinit();

    var component_count: usize = 0;
    for (0..graph.vertexCount()) |root| {
        if (ids[root] != std.math.maxInt(usize)) continue;

        ids[root] = component_count;
        try queue.append(root);
        var cursor: usize = 0;
        while (cursor < queue.items.len) : (cursor += 1) {
            const current = queue.items[cursor];
            for (0..graph.vertexCount()) |neighbor| {
                if (!graph.adjacent(current, neighbor)) continue;
                if (ids[neighbor] != std.math.maxInt(usize)) continue;
                ids[neighbor] = component_count;
                try queue.append(neighbor);
            }
        }
        queue.clearRetainingCapacity();
        component_count += 1;
    }

    return .{
        .allocator = allocator,
        .component_ids = ids,
        .component_count = component_count,
    };
}

pub fn pathWitness(
    allocator: std.mem.Allocator,
    complex: topology.Complex,
    source: topology.Vertex,
    target: topology.Vertex,
) !OwnedVertexPath {
    var graph = try oneSkeletonAdjacency(allocator, complex);
    defer graph.deinit();

    const source_index = indexOfVertex(graph.vertices, source) orelse return error.SourceVertexOutsideComplex;
    const target_index = indexOfVertex(graph.vertices, target) orelse return error.TargetVertexOutsideComplex;

    const previous = try allocator.alloc(?usize, graph.vertexCount());
    defer allocator.free(previous);
    @memset(previous, null);

    const seen = try allocator.alloc(bool, graph.vertexCount());
    defer allocator.free(seen);
    @memset(seen, false);

    var queue = std.ArrayList(usize).init(allocator);
    defer queue.deinit();
    seen[source_index] = true;
    try queue.append(source_index);

    var cursor: usize = 0;
    while (cursor < queue.items.len) : (cursor += 1) {
        const current = queue.items[cursor];
        if (current == target_index) break;

        for (0..graph.vertexCount()) |neighbor| {
            if (!graph.adjacent(current, neighbor)) continue;
            if (seen[neighbor]) continue;
            seen[neighbor] = true;
            previous[neighbor] = current;
            try queue.append(neighbor);
        }
    }

    if (!seen[target_index]) return error.NoPath;

    var reversed = std.ArrayList(topology.Vertex).init(allocator);
    defer reversed.deinit();

    var current: ?usize = target_index;
    while (current) |index| {
        try reversed.append(graph.vertices[index]);
        current = previous[index];
    }

    const path = try allocator.alloc(topology.Vertex, reversed.items.len);
    for (reversed.items, 0..) |vertex, index| {
        path[path.len - 1 - index] = vertex;
    }

    return .{ .allocator = allocator, .vertices = path };
}

pub fn bettiNumbers(
    allocator: std.mem.Allocator,
    complex: topology.Complex,
    max_dimension: usize,
    prime: u16,
) !OwnedBettiNumbers {
    if (!isPrime(prime)) return error.InvalidFieldPrime;
    try complex.validate();

    var faces = try complex.faces(allocator);
    defer faces.deinit();

    const values = try allocator.alloc(usize, max_dimension + 1);
    errdefer allocator.free(values);

    var rank_boundary_k = try allocator.alloc(usize, max_dimension + 2);
    defer allocator.free(rank_boundary_k);
    @memset(rank_boundary_k, 0);

    for (1..max_dimension + 2) |dimension| {
        rank_boundary_k[dimension] = try boundaryRank(allocator, faces.complex(), dimension, prime);
    }

    for (0..max_dimension + 1) |dimension| {
        const simplex_count = countSimplexesOfDimension(faces.complex(), @intCast(dimension));
        const cycles = simplex_count - rank_boundary_k[dimension];
        values[dimension] = cycles - rank_boundary_k[dimension + 1];
    }

    return .{ .allocator = allocator, .values = values };
}

pub fn checkBettiPreservingMap(
    allocator: std.mem.Allocator,
    map: topology.VertexMap,
    max_dimension: usize,
    prime: u16,
) !HomologyMapReport {
    try map.validate(allocator, .{});

    var image = try map.image(allocator);
    defer image.deinit();

    var source = try bettiNumbers(allocator, map.domain, max_dimension, prime);
    defer source.deinit();
    var image_betti = try bettiNumbers(allocator, image.complex(), max_dimension, prime);
    defer image_betti.deinit();

    const source_values = try allocator.dupe(usize, source.values);
    errdefer allocator.free(source_values);
    const image_values = try allocator.dupe(usize, image_betti.values);
    errdefer allocator.free(image_values);

    const induced_ranks = try allocator.alloc(usize, max_dimension + 1);
    errdefer allocator.free(induced_ranks);
    var preserves_homology = true;
    for (0..max_dimension + 1) |dimension| {
        induced_ranks[dimension] = try inducedHomologyRank(
            allocator,
            .{ .domain = map.domain, .codomain = image.complex(), .entries = map.entries },
            dimension,
            prime,
        );
        if (induced_ranks[dimension] != source_values[dimension] or
            induced_ranks[dimension] != image_values[dimension])
        {
            preserves_homology = false;
        }
    }

    return .{
        .allocator = allocator,
        .source_betti = source_values,
        .image_betti = image_values,
        .induced_ranks = induced_ranks,
        .preserves_betti = std.mem.eql(usize, source_values, image_values),
        .preserves_homology = preserves_homology,
    };
}

pub fn findHomologyMapWitness(
    allocator: std.mem.Allocator,
    map: topology.VertexMap,
    max_dimension: usize,
    prime: u16,
) !?OwnedHomologyMapWitness {
    if (!isPrime(prime)) return error.InvalidFieldPrime;
    try map.validate(allocator, .{});

    var image = try map.image(allocator);
    defer image.deinit();

    var source_faces = try map.domain.faces(allocator);
    defer source_faces.deinit();
    var image_faces = try image.complex().faces(allocator);
    defer image_faces.deinit();

    var dimension: usize = 1;
    while (dimension <= max_dimension) : (dimension += 1) {
        var source_k = try simplexesOfDimension(allocator, source_faces.complex(), @intCast(dimension));
        defer source_k.deinit();
        var image_k = try simplexesOfDimension(allocator, image_faces.complex(), @intCast(dimension));
        defer image_k.deinit();

        if (source_k.simplexes.len == 0 and image_k.simplexes.len == 0) continue;

        var source_previous = try simplexesOfDimension(allocator, source_faces.complex(), @intCast(dimension - 1));
        defer source_previous.deinit();
        const source_boundary = try boundaryMatrixForSimplexes(
            allocator,
            source_previous.simplexes,
            source_k.simplexes,
            prime,
        );
        defer allocator.free(source_boundary);

        const source_cycles = try kernelBasisModPrime(
            allocator,
            source_boundary,
            source_previous.simplexes.len,
            source_k.simplexes.len,
            prime,
        );
        defer allocator.free(source_cycles);
        const source_cycle_count = if (source_k.simplexes.len == 0) 0 else source_cycles.len / source_k.simplexes.len;

        var source_next = try simplexesOfDimension(allocator, source_faces.complex(), @intCast(dimension + 1));
        defer source_next.deinit();
        const source_boundaries = try boundaryMatrixForSimplexes(
            allocator,
            source_k.simplexes,
            source_next.simplexes,
            prime,
        );
        defer allocator.free(source_boundaries);

        var image_previous = try simplexesOfDimension(allocator, image_faces.complex(), @intCast(dimension - 1));
        defer image_previous.deinit();
        const image_boundary = try boundaryMatrixForSimplexes(
            allocator,
            image_previous.simplexes,
            image_k.simplexes,
            prime,
        );
        defer allocator.free(image_boundary);

        const image_cycles = try kernelBasisModPrime(
            allocator,
            image_boundary,
            image_previous.simplexes.len,
            image_k.simplexes.len,
            prime,
        );
        defer allocator.free(image_cycles);
        const image_cycle_count = if (image_k.simplexes.len == 0) 0 else image_cycles.len / image_k.simplexes.len;

        var image_next = try simplexesOfDimension(allocator, image_faces.complex(), @intCast(dimension + 1));
        defer image_next.deinit();
        const image_boundaries = try boundaryMatrixForSimplexes(
            allocator,
            image_k.simplexes,
            image_next.simplexes,
            prime,
        );
        defer allocator.free(image_boundaries);

        var source_cycle_index: usize = 0;
        while (source_cycle_index < source_cycle_count) : (source_cycle_index += 1) {
            const source_cycle = source_cycles[source_cycle_index * source_k.simplexes.len ..][0..source_k.simplexes.len];
            if (try chainInColumnSpan(
                allocator,
                source_boundaries,
                source_k.simplexes.len,
                source_next.simplexes.len,
                source_cycle,
                prime,
            )) continue;

            const mapped_cycle = try mapCycleToTargetChain(
                allocator,
                map,
                source_k.simplexes,
                source_cycle,
                image_k.simplexes,
                prime,
            );
            defer allocator.free(mapped_cycle);

            if (try chainInColumnSpan(
                allocator,
                image_boundaries,
                image_k.simplexes.len,
                image_next.simplexes.len,
                mapped_cycle,
                prime,
            )) {
                var source_chain = try chainFromVector(
                    allocator,
                    dimension,
                    prime,
                    source_k.simplexes,
                    source_cycle,
                );
                errdefer source_chain.deinit();
                var image_chain = try chainFromVector(
                    allocator,
                    dimension,
                    prime,
                    image_k.simplexes,
                    mapped_cycle,
                );
                errdefer image_chain.deinit();
                return .{
                    .allocator = allocator,
                    .kind = .source_cycle_lost,
                    .dimension = dimension,
                    .prime = prime,
                    .source_cycle = source_chain,
                    .image_cycle = image_chain,
                };
            }
        }

        const mapped_columns = try mappedSourceCycleColumns(
            allocator,
            map,
            source_k.simplexes,
            source_cycles,
            source_cycle_count,
            image_k.simplexes,
            prime,
        );
        defer allocator.free(mapped_columns);

        const span_columns = source_cycle_count + image_next.simplexes.len;
        const span = try allocator.alloc(u16, image_k.simplexes.len * span_columns);
        defer allocator.free(span);
        @memset(span, 0);
        copyMatrixColumns(
            span,
            span_columns,
            0,
            mapped_columns,
            source_cycle_count,
            image_k.simplexes.len,
            source_cycle_count,
        );
        copyMatrixColumns(
            span,
            span_columns,
            source_cycle_count,
            image_boundaries,
            image_next.simplexes.len,
            image_k.simplexes.len,
            image_next.simplexes.len,
        );

        var image_cycle_index: usize = 0;
        while (image_cycle_index < image_cycle_count) : (image_cycle_index += 1) {
            const image_cycle = image_cycles[image_cycle_index * image_k.simplexes.len ..][0..image_k.simplexes.len];
            if (try chainInColumnSpan(
                allocator,
                image_boundaries,
                image_k.simplexes.len,
                image_next.simplexes.len,
                image_cycle,
                prime,
            )) continue;

            if (!(try chainInColumnSpan(
                allocator,
                span,
                image_k.simplexes.len,
                span_columns,
                image_cycle,
                prime,
            ))) {
                var image_chain = try chainFromVector(
                    allocator,
                    dimension,
                    prime,
                    image_k.simplexes,
                    image_cycle,
                );
                errdefer image_chain.deinit();
                return .{
                    .allocator = allocator,
                    .kind = .image_cycle_created,
                    .dimension = dimension,
                    .prime = prime,
                    .image_cycle = image_chain,
                };
            }
        }
    }

    return null;
}

pub fn fundamentalGroupPresentation(
    allocator: std.mem.Allocator,
    complex: topology.Complex,
) !FundamentalGroupPresentation {
    if (complex.dimension() > 2) return error.ComplexDimensionTooHigh;

    var faces = try complex.faces(allocator);
    defer faces.deinit();
    var vertices = try complex.vertices(allocator);
    defer vertices.deinit();

    var edges = std.ArrayList(topology.Simplex).init(allocator);
    defer edges.deinit();
    var triangles = std.ArrayList(topology.Simplex).init(allocator);
    defer triangles.deinit();
    for (faces.simplexes) |simplex| {
        if (simplex.dimension() == 1) try edges.append(simplex);
        if (simplex.dimension() == 2) try triangles.append(simplex);
    }

    const tree_edges = try spanningForestEdges(allocator, vertices.vertices, edges.items);
    defer allocator.free(tree_edges);

    var generators = std.ArrayList(EdgeGenerator).init(allocator);
    errdefer generators.deinit();
    for (edges.items, 0..) |edge, edge_index| {
        if (tree_edges[edge_index]) continue;
        var ordered = try orderedEdge(allocator, edge);
        defer ordered.deinit();
        try generators.append(.{
            .name = generators.items.len,
            .from = ordered.vertices[0],
            .to = ordered.vertices[1],
        });
    }

    var relators = std.ArrayList(Relator).init(allocator);
    errdefer deinitRelatorList(allocator, &relators);
    for (triangles.items) |triangle| {
        var word = std.ArrayList(SignedGenerator).init(allocator);
        errdefer word.deinit();

        var ordered = try triangle.canonicalize(allocator);
        defer ordered.deinit();
        try appendBoundaryGenerator(&word, generators.items, ordered.vertices[0], ordered.vertices[1]);
        try appendBoundaryGenerator(&word, generators.items, ordered.vertices[1], ordered.vertices[2]);
        try appendBoundaryGenerator(&word, generators.items, ordered.vertices[0], ordered.vertices[2]);

        try relators.append(.{ .word = try word.toOwnedSlice() });
    }

    return .{
        .allocator = allocator,
        .generators = try generators.toOwnedSlice(),
        .relators = try relators.toOwnedSlice(),
    };
}

pub fn findConnectivityObstruction(
    allocator: std.mem.Allocator,
    execution: topology.SimplicialCarrierMap,
    task: topology.SimplicialCarrierMap,
) !?OwnedConnectivityObstructionReport {
    try execution.validate(allocator, .{});
    try task.validate(allocator, .{});

    var domain_faces = try execution.domain.faces(allocator);
    defer domain_faces.deinit();

    for (domain_faces.simplexes) |input_simplex| {
        if (input_simplex.isEmpty()) continue;

        const protocol_image = execution.imageOf(input_simplex) orelse return error.MissingCarrierSimplex;
        const output_image = task.imageOf(input_simplex) orelse return error.MissingCarrierSimplex;

        var protocol_components = try connectedComponents(allocator, protocol_image.complex());
        defer protocol_components.deinit();
        var output_components = try connectedComponents(allocator, output_image.complex());
        defer output_components.deinit();

        if (protocol_components.component_count == 1 and output_components.component_count > 1) {
            return .{
                .allocator = allocator,
                .kind = .output_disconnected_for_connected_protocol,
                .input_vertices = try allocator.dupe(topology.Vertex, input_simplex.vertices),
                .protocol_components = protocol_components.component_count,
                .output_components = output_components.component_count,
            };
        }
    }

    return null;
}

fn boundaryRank(
    allocator: std.mem.Allocator,
    faces: topology.Complex,
    dimension: usize,
    prime: u16,
) !usize {
    if (dimension == 0) return 0;

    const row_dimension: isize = @intCast(dimension - 1);
    const column_dimension: isize = @intCast(dimension);
    var rows = try simplexesOfDimension(allocator, faces, row_dimension);
    defer rows.deinit();
    var columns = try simplexesOfDimension(allocator, faces, column_dimension);
    defer columns.deinit();

    if (rows.simplexes.len == 0 or columns.simplexes.len == 0) return 0;

    const matrix = try boundaryMatrixForSimplexes(allocator, rows.simplexes, columns.simplexes, prime);
    defer allocator.free(matrix);

    return rankModPrime(matrix, rows.simplexes.len, columns.simplexes.len, prime);
}

fn boundaryMatrixForSimplexes(
    allocator: std.mem.Allocator,
    rows: []const topology.Simplex,
    columns: []const topology.Simplex,
    prime: u16,
) ![]u16 {
    const matrix = try allocator.alloc(u16, rows.len * columns.len);
    errdefer allocator.free(matrix);
    @memset(matrix, 0);

    if (rows.len == 0 or columns.len == 0) return matrix;

    for (columns, 0..) |column_simplex, column| {
        var canonical = try column_simplex.canonicalize(allocator);
        defer canonical.deinit();
        for (canonical.vertices, 0..) |_, removed_index| {
            const face_vertices = try allocator.alloc(topology.Vertex, canonical.vertices.len - 1);
            defer allocator.free(face_vertices);
            var out_index: usize = 0;
            for (canonical.vertices, 0..) |vertex, vertex_index| {
                if (vertex_index == removed_index) continue;
                face_vertices[out_index] = vertex;
                out_index += 1;
            }

            const row = indexOfSimplex(rows, .{ .vertices = face_vertices }) orelse {
                return error.MissingBoundaryFace;
            };
            const coefficient: u16 = if (removed_index % 2 == 0) 1 else prime - 1;
            matrix[row * columns.len + column] = coefficient;
        }
    }

    return matrix;
}

fn simplexesOfDimension(
    allocator: std.mem.Allocator,
    complex: topology.Complex,
    dimension: isize,
) !topology.OwnedSimplexSet {
    var builder = topology.ComplexBuilder.init(allocator);
    errdefer builder.deinit();
    for (complex.simplexes) |simplex| {
        if (simplex.dimension() == dimension) try builder.add(simplex);
    }
    return builder.toOwnedSet();
}

fn countSimplexesOfDimension(complex: topology.Complex, dimension: isize) usize {
    var count: usize = 0;
    for (complex.simplexes) |simplex| {
        if (simplex.dimension() == dimension) count += 1;
    }
    return count;
}

fn inducedHomologyRank(
    allocator: std.mem.Allocator,
    map: topology.VertexMap,
    dimension: usize,
    prime: u16,
) !usize {
    if (!isPrime(prime)) return error.InvalidFieldPrime;
    try map.validate(allocator, .{});

    var source_faces = try map.domain.faces(allocator);
    defer source_faces.deinit();
    var target_faces = try map.codomain.faces(allocator);
    defer target_faces.deinit();

    var source_k = try simplexesOfDimension(allocator, source_faces.complex(), @intCast(dimension));
    defer source_k.deinit();
    var target_k = try simplexesOfDimension(allocator, target_faces.complex(), @intCast(dimension));
    defer target_k.deinit();

    var source_boundary: []u16 = undefined;
    var source_boundary_rows: usize = 0;
    if (dimension == 0) {
        source_boundary = try allocator.alloc(u16, 0);
    } else {
        var source_previous = try simplexesOfDimension(allocator, source_faces.complex(), @intCast(dimension - 1));
        defer source_previous.deinit();
        source_boundary_rows = source_previous.simplexes.len;
        source_boundary = try boundaryMatrixForSimplexes(allocator, source_previous.simplexes, source_k.simplexes, prime);
    }
    defer allocator.free(source_boundary);

    const source_cycles = try kernelBasisModPrime(
        allocator,
        source_boundary,
        source_boundary_rows,
        source_k.simplexes.len,
        prime,
    );
    defer allocator.free(source_cycles);
    const cycle_count = if (source_k.simplexes.len == 0) 0 else source_cycles.len / source_k.simplexes.len;

    var target_next = try simplexesOfDimension(allocator, target_faces.complex(), @intCast(dimension + 1));
    defer target_next.deinit();
    const target_boundaries = try boundaryMatrixForSimplexes(allocator, target_k.simplexes, target_next.simplexes, prime);
    defer allocator.free(target_boundaries);
    const target_boundary_rank = try rankModPrimeCopy(
        allocator,
        target_boundaries,
        target_k.simplexes.len,
        target_next.simplexes.len,
        prime,
    );

    const combined_columns = cycle_count + target_next.simplexes.len;
    if (target_k.simplexes.len == 0 or combined_columns == 0) return 0;

    const combined = try allocator.alloc(u16, target_k.simplexes.len * combined_columns);
    defer allocator.free(combined);
    @memset(combined, 0);

    for (0..cycle_count) |cycle_index| {
        const cycle = source_cycles[cycle_index * source_k.simplexes.len ..][0..source_k.simplexes.len];
        try appendMappedCycleColumn(
            allocator,
            combined,
            combined_columns,
            cycle_index,
            target_k.simplexes,
            map,
            source_k.simplexes,
            cycle,
            prime,
        );
    }

    for (target_next.simplexes, 0..) |_, boundary_column| {
        for (target_k.simplexes, 0..) |_, row| {
            combined[row * combined_columns + cycle_count + boundary_column] =
                target_boundaries[row * target_next.simplexes.len + boundary_column];
        }
    }

    const combined_rank = try rankModPrime(combined, target_k.simplexes.len, combined_columns, prime);
    return combined_rank - target_boundary_rank;
}

fn appendMappedCycleColumn(
    allocator: std.mem.Allocator,
    matrix: []u16,
    column_count: usize,
    column: usize,
    target_simplexes: []const topology.Simplex,
    map: topology.VertexMap,
    source_simplexes: []const topology.Simplex,
    cycle: []const u16,
    prime: u16,
) !void {
    for (source_simplexes, 0..) |source_simplex, source_index| {
        const source_coefficient = cycle[source_index] % prime;
        if (source_coefficient == 0) continue;

        const mapped = try mappedSimplexColumn(allocator, map, source_simplex, target_simplexes, prime) orelse continue;
        const contribution = modMul(source_coefficient, mapped.coefficient, prime);
        const matrix_index = mapped.row * column_count + column;
        matrix[matrix_index] = modAdd(matrix[matrix_index], contribution, prime);
    }
}

const MappedSimplexColumn = struct {
    row: usize,
    coefficient: u16,
};

fn mappedSimplexColumn(
    allocator: std.mem.Allocator,
    map: topology.VertexMap,
    source_simplex: topology.Simplex,
    target_simplexes: []const topology.Simplex,
    prime: u16,
) !?MappedSimplexColumn {
    const mapped_vertices = try allocator.alloc(topology.Vertex, source_simplex.vertices.len);
    defer allocator.free(mapped_vertices);

    for (source_simplex.vertices, 0..) |source, index| {
        const target = map.targetForVertex(source) orelse return error.MissingVertexMapEntry;
        for (mapped_vertices[0..index]) |existing| {
            if (topology.Vertex.eql(existing, target)) return null;
        }
        mapped_vertices[index] = target;
    }

    const row = indexOfSimplex(target_simplexes, .{ .vertices = mapped_vertices }) orelse return error.NonSimplicialMap;
    const odd = try permutationIsOdd(mapped_vertices, target_simplexes[row].vertices);
    return .{
        .row = row,
        .coefficient = if (odd) prime - 1 else 1,
    };
}

fn mapCycleToTargetChain(
    allocator: std.mem.Allocator,
    map: topology.VertexMap,
    source_simplexes: []const topology.Simplex,
    source_cycle: []const u16,
    target_simplexes: []const topology.Simplex,
    prime: u16,
) ![]u16 {
    const target_cycle = try allocator.alloc(u16, target_simplexes.len);
    errdefer allocator.free(target_cycle);
    @memset(target_cycle, 0);

    for (source_simplexes, 0..) |source_simplex, source_index| {
        const coefficient = source_cycle[source_index] % prime;
        if (coefficient == 0) continue;

        const mapped = try mappedSimplexColumn(allocator, map, source_simplex, target_simplexes, prime) orelse continue;
        const contribution = modMul(coefficient, mapped.coefficient, prime);
        target_cycle[mapped.row] = modAdd(target_cycle[mapped.row], contribution, prime);
    }

    return target_cycle;
}

fn mappedSourceCycleColumns(
    allocator: std.mem.Allocator,
    map: topology.VertexMap,
    source_simplexes: []const topology.Simplex,
    source_cycles: []const u16,
    source_cycle_count: usize,
    target_simplexes: []const topology.Simplex,
    prime: u16,
) ![]u16 {
    const matrix = try allocator.alloc(u16, target_simplexes.len * source_cycle_count);
    errdefer allocator.free(matrix);
    @memset(matrix, 0);

    var cycle_index: usize = 0;
    while (cycle_index < source_cycle_count) : (cycle_index += 1) {
        const source_cycle = source_cycles[cycle_index * source_simplexes.len ..][0..source_simplexes.len];
        const mapped = try mapCycleToTargetChain(allocator, map, source_simplexes, source_cycle, target_simplexes, prime);
        defer allocator.free(mapped);
        for (mapped, 0..) |coefficient, row| {
            matrix[row * source_cycle_count + cycle_index] = coefficient;
        }
    }

    return matrix;
}

fn copyMatrixColumns(
    destination: []u16,
    destination_columns: usize,
    destination_column_offset: usize,
    source: []const u16,
    source_columns: usize,
    rows: usize,
    copy_columns: usize,
) void {
    if (copy_columns == 0) return;
    for (0..rows) |row| {
        for (0..copy_columns) |column| {
            destination[row * destination_columns + destination_column_offset + column] =
                source[row * source_columns + column];
        }
    }
}

fn chainInColumnSpan(
    allocator: std.mem.Allocator,
    matrix: []const u16,
    rows: usize,
    columns: usize,
    vector: []const u16,
    prime: u16,
) !bool {
    if (vector.len != rows) return error.MatrixVectorMismatch;

    const rank_before = try rankModPrimeCopy(allocator, matrix, rows, columns, prime);
    const augmented_columns = columns + 1;
    const augmented = try allocator.alloc(u16, rows * augmented_columns);
    defer allocator.free(augmented);

    for (0..rows) |row| {
        for (0..columns) |column| {
            augmented[row * augmented_columns + column] = matrix[row * columns + column];
        }
        augmented[row * augmented_columns + columns] = vector[row] % prime;
    }

    const rank_after = try rankModPrime(augmented, rows, augmented_columns, prime);
    return rank_before == rank_after;
}

fn chainFromVector(
    allocator: std.mem.Allocator,
    dimension: usize,
    prime: u16,
    simplexes: []const topology.Simplex,
    vector: []const u16,
) !OwnedChain {
    var simplex_items = std.ArrayList(topology.Simplex).init(allocator);
    var coefficients = std.ArrayList(u16).init(allocator);
    errdefer {
        for (simplex_items.items) |simplex| allocator.free(@constCast(simplex.vertices));
        simplex_items.deinit();
        coefficients.deinit();
    }

    for (simplexes, 0..) |simplex, index| {
        const coefficient = vector[index] % prime;
        if (coefficient == 0) continue;

        var canonical = try simplex.canonicalize(allocator);
        errdefer canonical.deinit();
        try simplex_items.append(canonical.simplex());
        canonical.vertices = &.{};
        try coefficients.append(coefficient);
    }

    const owned_simplexes = try simplex_items.toOwnedSlice();
    errdefer {
        for (owned_simplexes) |simplex| allocator.free(@constCast(simplex.vertices));
        allocator.free(owned_simplexes);
    }
    const owned_coefficients = try coefficients.toOwnedSlice();

    return .{
        .allocator = allocator,
        .dimension = dimension,
        .prime = prime,
        .simplexes = owned_simplexes,
        .coefficients = owned_coefficients,
    };
}

fn permutationIsOdd(order: []const topology.Vertex, canonical: []const topology.Vertex) !bool {
    var inversions: usize = 0;
    for (order, 0..) |left, left_index| {
        const left_position = indexOfVertex(canonical, left) orelse return error.VertexNotFound;
        for (order[left_index + 1 ..]) |right| {
            const right_position = indexOfVertex(canonical, right) orelse return error.VertexNotFound;
            if (left_position > right_position) inversions += 1;
        }
    }
    return inversions % 2 == 1;
}

fn kernelBasisModPrime(
    allocator: std.mem.Allocator,
    matrix: []const u16,
    rows: usize,
    columns: usize,
    prime: u16,
) ![]u16 {
    if (columns == 0) return allocator.alloc(u16, 0);

    const rref = try allocator.dupe(u16, matrix);
    defer allocator.free(rref);
    const pivots = try rrefPivotColumns(allocator, rref, rows, columns, prime);
    defer allocator.free(pivots);

    const is_pivot = try allocator.alloc(bool, columns);
    defer allocator.free(is_pivot);
    @memset(is_pivot, false);
    for (pivots) |pivot| {
        is_pivot[pivot] = true;
    }

    var free_count: usize = 0;
    for (is_pivot) |pivot| {
        if (!pivot) free_count += 1;
    }

    const basis = try allocator.alloc(u16, free_count * columns);
    errdefer allocator.free(basis);
    @memset(basis, 0);

    var basis_index: usize = 0;
    for (0..columns) |free_column| {
        if (is_pivot[free_column]) continue;

        const vector = basis[basis_index * columns ..][0..columns];
        vector[free_column] = 1;
        for (pivots, 0..) |pivot_column, pivot_row| {
            const coefficient = rref[pivot_row * columns + free_column] % prime;
            vector[pivot_column] = if (coefficient == 0) 0 else prime - coefficient;
        }
        basis_index += 1;
    }

    return basis;
}

fn rankModPrimeCopy(
    allocator: std.mem.Allocator,
    matrix: []const u16,
    rows: usize,
    columns: usize,
    prime: u16,
) !usize {
    const copy = try allocator.dupe(u16, matrix);
    defer allocator.free(copy);
    return rankModPrime(copy, rows, columns, prime);
}

fn rrefPivotColumns(
    allocator: std.mem.Allocator,
    matrix: []u16,
    rows: usize,
    columns: usize,
    prime: u16,
) ![]usize {
    var pivots = std.ArrayList(usize).init(allocator);
    errdefer pivots.deinit();

    var rank: usize = 0;
    var column: usize = 0;
    while (column < columns and rank < rows) : (column += 1) {
        var pivot: ?usize = null;
        var row: usize = rank;
        while (row < rows) : (row += 1) {
            if (matrix[row * columns + column] % prime != 0) {
                pivot = row;
                break;
            }
        }
        if (pivot == null) continue;

        if (pivot.? != rank) {
            swapRows(matrix, columns, pivot.?, rank);
        }

        const inverse = try inverseModPrime(matrix[rank * columns + column], prime);
        for (column..columns) |entry_column| {
            matrix[rank * columns + entry_column] = modMul(matrix[rank * columns + entry_column], inverse, prime);
        }

        for (0..rows) |elim_row| {
            if (elim_row == rank) continue;
            const factor = matrix[elim_row * columns + column] % prime;
            if (factor == 0) continue;
            for (column..columns) |entry_column| {
                const scaled = modMul(factor, matrix[rank * columns + entry_column], prime);
                matrix[elim_row * columns + entry_column] = modSub(matrix[elim_row * columns + entry_column], scaled, prime);
            }
        }

        try pivots.append(column);
        rank += 1;
    }

    return pivots.toOwnedSlice();
}

fn rankModPrime(matrix: []u16, rows: usize, columns: usize, prime: u16) !usize {
    var rank: usize = 0;
    var column: usize = 0;
    while (column < columns and rank < rows) : (column += 1) {
        var pivot: ?usize = null;
        var row: usize = rank;
        while (row < rows) : (row += 1) {
            if (matrix[row * columns + column] % prime != 0) {
                pivot = row;
                break;
            }
        }
        if (pivot == null) continue;

        if (pivot.? != rank) {
            swapRows(matrix, columns, pivot.?, rank);
        }

        const inverse = try inverseModPrime(matrix[rank * columns + column], prime);
        for (column..columns) |entry_column| {
            matrix[rank * columns + entry_column] = modMul(matrix[rank * columns + entry_column], inverse, prime);
        }

        for (0..rows) |elim_row| {
            if (elim_row == rank) continue;
            const factor = matrix[elim_row * columns + column] % prime;
            if (factor == 0) continue;
            for (column..columns) |entry_column| {
                const scaled = modMul(factor, matrix[rank * columns + entry_column], prime);
                matrix[elim_row * columns + entry_column] = modSub(matrix[elim_row * columns + entry_column], scaled, prime);
            }
        }

        rank += 1;
    }
    return rank;
}

fn swapRows(matrix: []u16, columns: usize, left: usize, right: usize) void {
    for (0..columns) |column| {
        const left_index = left * columns + column;
        const right_index = right * columns + column;
        const tmp = matrix[left_index];
        matrix[left_index] = matrix[right_index];
        matrix[right_index] = tmp;
    }
}

fn inverseModPrime(value: u16, prime: u16) !u16 {
    const normalized = value % prime;
    if (normalized == 0) return error.NoModularInverse;
    var candidate: u16 = 1;
    while (candidate < prime) : (candidate += 1) {
        if (modMul(normalized, candidate, prime) == 1) return candidate;
    }
    return error.NoModularInverse;
}

fn modAdd(left: u16, right: u16, prime: u16) u16 {
    return @intCast((@as(u32, left % prime) + @as(u32, right % prime)) % @as(u32, prime));
}

fn modMul(left: u16, right: u16, prime: u16) u16 {
    return @intCast((@as(u32, left % prime) * @as(u32, right % prime)) % @as(u32, prime));
}

fn modSub(left: u16, right: u16, prime: u16) u16 {
    return @intCast((@as(u32, left % prime) + @as(u32, prime) - @as(u32, right % prime)) % @as(u32, prime));
}

fn isPrime(value: u16) bool {
    if (value < 2) return false;
    var divisor: u16 = 2;
    while (@as(u32, divisor) * @as(u32, divisor) <= @as(u32, value)) : (divisor += 1) {
        if (value % divisor == 0) return false;
    }
    return true;
}

fn spanningForestEdges(
    allocator: std.mem.Allocator,
    vertices: []const topology.Vertex,
    edges: []const topology.Simplex,
) ![]bool {
    const in_tree = try allocator.alloc(bool, edges.len);
    errdefer allocator.free(in_tree);
    @memset(in_tree, false);

    const parent = try allocator.alloc(usize, vertices.len);
    defer allocator.free(parent);
    for (parent, 0..) |*slot, index| {
        slot.* = index;
    }

    for (edges, 0..) |edge, edge_index| {
        var ordered = try orderedEdge(allocator, edge);
        defer ordered.deinit();

        const left = indexOfVertex(vertices, ordered.vertices[0]) orelse return error.VertexNotFound;
        const right = indexOfVertex(vertices, ordered.vertices[1]) orelse return error.VertexNotFound;
        const left_root = findSet(parent, left);
        const right_root = findSet(parent, right);
        if (left_root == right_root) continue;

        parent[right_root] = left_root;
        in_tree[edge_index] = true;
    }

    return in_tree;
}

fn findSet(parent: []usize, index: usize) usize {
    var current = index;
    while (parent[current] != current) {
        current = parent[current];
    }
    return current;
}

fn orderedEdge(allocator: std.mem.Allocator, edge: topology.Simplex) !topology.OwnedSimplex {
    if (edge.vertices.len != 2) return error.NotAnEdge;
    return edge.canonicalize(allocator);
}

fn appendBoundaryGenerator(
    word: *std.ArrayList(SignedGenerator),
    generators: []const EdgeGenerator,
    from: topology.Vertex,
    to: topology.Vertex,
) !void {
    for (generators, 0..) |generator, index| {
        if (topology.Vertex.eql(generator.from, from) and topology.Vertex.eql(generator.to, to)) {
            try word.append(.{ .generator_index = index, .inverse = false });
            return;
        }
        if (topology.Vertex.eql(generator.from, to) and topology.Vertex.eql(generator.to, from)) {
            try word.append(.{ .generator_index = index, .inverse = true });
            return;
        }
    }
}

fn deinitRelatorList(allocator: std.mem.Allocator, relators: *std.ArrayList(Relator)) void {
    for (relators.items) |relator| {
        allocator.free(relator.word);
    }
    relators.deinit();
}

fn indexOfVertex(vertices: []const topology.Vertex, needle: topology.Vertex) ?usize {
    for (vertices, 0..) |vertex, index| {
        if (topology.Vertex.eql(vertex, needle)) return index;
    }
    return null;
}

fn indexOfSimplex(simplexes: []const topology.Simplex, needle: topology.Simplex) ?usize {
    for (simplexes, 0..) |simplex, index| {
        if (topology.Simplex.eql(simplex, needle)) return index;
    }
    return null;
}

test "one skeleton adjacency components and path witnesses" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = topology.Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const ab = [_]topology.Vertex{ a, b };
    const bc = [_]topology.Vertex{ b, c };
    const complex = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &ab },
        .{ .vertices = &bc },
    } };

    var graph = try oneSkeletonAdjacency(std.testing.allocator, complex);
    defer graph.deinit();
    var components = try connectedComponents(std.testing.allocator, complex);
    defer components.deinit();
    var path = try pathWitness(std.testing.allocator, complex, a, c);
    defer path.deinit();

    try std.testing.expectEqual(@as(usize, 3), graph.vertexCount());
    try std.testing.expectEqual(@as(usize, 1), components.component_count);
    try std.testing.expectEqual(@as(usize, 3), path.vertices.len);
    try std.testing.expect(topology.Vertex.eql(a, path.vertices[0]));
    try std.testing.expect(topology.Vertex.eql(c, path.vertices[2]));
}

test "components detect disconnected one skeletons" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const a_vertex = [_]topology.Vertex{a};
    const b_vertex = [_]topology.Vertex{b};
    const complex = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &a_vertex },
        .{ .vertices = &b_vertex },
    } };

    var components = try connectedComponents(std.testing.allocator, complex);
    defer components.deinit();

    try std.testing.expectEqual(@as(usize, 2), components.component_count);
    try std.testing.expectError(error.NoPath, pathWitness(std.testing.allocator, complex, a, b));
}

test "k skeleton extraction returns the requested finite subcomplex" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = topology.Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const triangle = [_]topology.Vertex{ a, b, c };
    const complex = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &triangle }} };

    var skeleton = try kSkeleton(std.testing.allocator, complex, 1);
    defer skeleton.deinit();

    try std.testing.expectEqual(@as(isize, 1), skeleton.complex().dimension());
    try std.testing.expect(skeleton.complex().containsSimplex(.{ .vertices = &[_]topology.Vertex{ a, b } }));
    try std.testing.expect(!skeleton.complex().containsSimplex(.{ .vertices = &triangle }));
}

test "betti numbers distinguish interval circle and filled triangle" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = topology.Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const ab = [_]topology.Vertex{ a, b };
    const bc = [_]topology.Vertex{ b, c };
    const ac = [_]topology.Vertex{ a, c };
    const triangle = [_]topology.Vertex{ a, b, c };

    var interval = try bettiNumbers(std.testing.allocator, .{
        .simplexes = &[_]topology.Simplex{.{ .vertices = &ab }},
    }, 1, 2);
    defer interval.deinit();
    var circle = try bettiNumbers(std.testing.allocator, .{
        .simplexes = &[_]topology.Simplex{
            .{ .vertices = &ab },
            .{ .vertices = &bc },
            .{ .vertices = &ac },
        },
    }, 2, 2);
    defer circle.deinit();
    var filled = try bettiNumbers(std.testing.allocator, .{
        .simplexes = &[_]topology.Simplex{.{ .vertices = &triangle }},
    }, 2, 2);
    defer filled.deinit();

    try std.testing.expectEqual(@as(usize, 1), interval.values[0]);
    try std.testing.expectEqual(@as(usize, 0), interval.values[1]);
    try std.testing.expectEqual(@as(usize, 1), circle.values[0]);
    try std.testing.expectEqual(@as(usize, 1), circle.values[1]);
    try std.testing.expectEqual(@as(usize, 1), filled.values[0]);
    try std.testing.expectEqual(@as(usize, 0), filled.values[1]);
    try std.testing.expectEqual(@as(usize, 0), filled.values[2]);
}

test "betti numbers require a prime field" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const a_vertex = [_]topology.Vertex{a};
    const complex = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &a_vertex }} };

    try std.testing.expectError(error.InvalidFieldPrime, bettiNumbers(std.testing.allocator, complex, 0, 1));
    try std.testing.expectError(error.InvalidFieldPrime, bettiNumbers(std.testing.allocator, complex, 0, 4));
}

test "homology map report detects cycle collapse" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = topology.Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const point = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const ab = [_]topology.Vertex{ a, b };
    const bc = [_]topology.Vertex{ b, c };
    const ac = [_]topology.Vertex{ a, c };
    const point_vertices = [_]topology.Vertex{point};
    const domain = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &ab },
        .{ .vertices = &bc },
        .{ .vertices = &ac },
    } };
    const codomain = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &point_vertices }} };
    const entries = [_]topology.VertexMapEntry{
        .{ .source = a, .target = point },
        .{ .source = b, .target = point },
        .{ .source = c, .target = point },
    };

    var report = try checkBettiPreservingMap(std.testing.allocator, .{
        .domain = domain,
        .codomain = codomain,
        .entries = &entries,
    }, 1, 2);
    defer report.deinit();

    try std.testing.expect(!report.preserves_betti);
    try std.testing.expect(!report.preserves_homology);
    try std.testing.expectEqual(@as(usize, 1), report.source_betti[1]);
    try std.testing.expectEqual(@as(usize, 0), report.image_betti[1]);
    try std.testing.expectEqual(@as(usize, 0), report.induced_ranks[1]);
}

test "homology map report detects same-betti cycle cancellation over a field" {
    const a0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const a1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const a2 = topology.Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const a3 = topology.Vertex{ .color = .{ .process = 3 }, .label = .{ .input = 3 } };
    const a4 = topology.Vertex{ .color = .{ .process = 4 }, .label = .{ .input = 4 } };
    const a5 = topology.Vertex{ .color = .{ .process = 5 }, .label = .{ .input = 5 } };
    const x0 = topology.Vertex{ .color = .{ .process = 10 }, .label = .{ .output = 0 } };
    const x1 = topology.Vertex{ .color = .{ .process = 11 }, .label = .{ .output = 1 } };
    const x2 = topology.Vertex{ .color = .{ .process = 12 }, .label = .{ .output = 2 } };

    const a01 = [_]topology.Vertex{ a0, a1 };
    const a12 = [_]topology.Vertex{ a1, a2 };
    const a23 = [_]topology.Vertex{ a2, a3 };
    const a34 = [_]topology.Vertex{ a3, a4 };
    const a45 = [_]topology.Vertex{ a4, a5 };
    const a50 = [_]topology.Vertex{ a5, a0 };
    const x01 = [_]topology.Vertex{ x0, x1 };
    const x12 = [_]topology.Vertex{ x1, x2 };
    const x20 = [_]topology.Vertex{ x2, x0 };
    const domain = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &a01 },
        .{ .vertices = &a12 },
        .{ .vertices = &a23 },
        .{ .vertices = &a34 },
        .{ .vertices = &a45 },
        .{ .vertices = &a50 },
    } };
    const codomain = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &x01 },
        .{ .vertices = &x12 },
        .{ .vertices = &x20 },
    } };
    const entries = [_]topology.VertexMapEntry{
        .{ .source = a0, .target = x0 },
        .{ .source = a1, .target = x1 },
        .{ .source = a2, .target = x2 },
        .{ .source = a3, .target = x0 },
        .{ .source = a4, .target = x1 },
        .{ .source = a5, .target = x2 },
    };

    var report = try checkBettiPreservingMap(std.testing.allocator, .{
        .domain = domain,
        .codomain = codomain,
        .entries = &entries,
    }, 1, 2);
    defer report.deinit();

    try std.testing.expect(report.preserves_betti);
    try std.testing.expect(!report.preserves_homology);
    try std.testing.expectEqual(@as(usize, 1), report.source_betti[1]);
    try std.testing.expectEqual(@as(usize, 1), report.image_betti[1]);
    try std.testing.expectEqual(@as(usize, 0), report.induced_ranks[1]);
}

test "homology map witness identifies a bounded source cycle lost by a projection" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = topology.Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const point = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const ab = [_]topology.Vertex{ a, b };
    const bc = [_]topology.Vertex{ b, c };
    const ac = [_]topology.Vertex{ a, c };
    const point_vertices = [_]topology.Vertex{point};
    const domain = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &ab },
        .{ .vertices = &bc },
        .{ .vertices = &ac },
    } };
    const codomain = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &point_vertices }} };
    const entries = [_]topology.VertexMapEntry{
        .{ .source = a, .target = point },
        .{ .source = b, .target = point },
        .{ .source = c, .target = point },
    };

    var witness = (try findHomologyMapWitness(std.testing.allocator, .{
        .domain = domain,
        .codomain = codomain,
        .entries = &entries,
    }, 1, 2)).?;
    defer witness.deinit();

    try std.testing.expectEqual(HomologyMapWitnessKind.source_cycle_lost, witness.kind);
    try std.testing.expectEqual(@as(usize, 1), witness.dimension);
    try std.testing.expectEqual(@as(usize, 3), witness.source_cycle.?.simplexes.len);
    try std.testing.expectEqual(@as(usize, 0), witness.image_cycle.?.simplexes.len);
}

test "homology map witness identifies a bounded image cycle created by local identifications" {
    const a0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const a1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const a2 = topology.Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const a3 = topology.Vertex{ .color = .{ .process = 3 }, .label = .{ .input = 3 } };
    const x0 = topology.Vertex{ .color = .{ .process = 10 }, .label = .{ .output = 0 } };
    const x1 = topology.Vertex{ .color = .{ .process = 11 }, .label = .{ .output = 1 } };
    const x2 = topology.Vertex{ .color = .{ .process = 12 }, .label = .{ .output = 2 } };

    const a01 = [_]topology.Vertex{ a0, a1 };
    const a12 = [_]topology.Vertex{ a1, a2 };
    const a23 = [_]topology.Vertex{ a2, a3 };
    const x01 = [_]topology.Vertex{ x0, x1 };
    const x12 = [_]topology.Vertex{ x1, x2 };
    const x20 = [_]topology.Vertex{ x2, x0 };
    const domain = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &a01 },
        .{ .vertices = &a12 },
        .{ .vertices = &a23 },
    } };
    const codomain = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &x01 },
        .{ .vertices = &x12 },
        .{ .vertices = &x20 },
    } };
    const entries = [_]topology.VertexMapEntry{
        .{ .source = a0, .target = x0 },
        .{ .source = a1, .target = x1 },
        .{ .source = a2, .target = x2 },
        .{ .source = a3, .target = x0 },
    };

    var witness = (try findHomologyMapWitness(std.testing.allocator, .{
        .domain = domain,
        .codomain = codomain,
        .entries = &entries,
    }, 1, 2)).?;
    defer witness.deinit();

    try std.testing.expectEqual(HomologyMapWitnessKind.image_cycle_created, witness.kind);
    try std.testing.expectEqual(@as(usize, 1), witness.dimension);
    try std.testing.expectEqual(@as(usize, 3), witness.image_cycle.?.simplexes.len);
    try std.testing.expect(witness.source_cycle == null);
}

test "fundamental group presentation exposes cycle generators and triangle relators" {
    const a = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = topology.Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const ab = [_]topology.Vertex{ a, b };
    const bc = [_]topology.Vertex{ b, c };
    const ac = [_]topology.Vertex{ a, c };
    const triangle = [_]topology.Vertex{ a, b, c };

    var circle = try fundamentalGroupPresentation(std.testing.allocator, .{
        .simplexes = &[_]topology.Simplex{
            .{ .vertices = &ab },
            .{ .vertices = &bc },
            .{ .vertices = &ac },
        },
    });
    defer circle.deinit();
    var filled = try fundamentalGroupPresentation(std.testing.allocator, .{
        .simplexes = &[_]topology.Simplex{.{ .vertices = &triangle }},
    });
    defer filled.deinit();

    try std.testing.expectEqual(@as(usize, 1), circle.generators.len);
    try std.testing.expectEqual(@as(usize, 0), circle.relators.len);
    try std.testing.expectEqual(@as(usize, 1), filled.generators.len);
    try std.testing.expectEqual(@as(usize, 1), filled.relators.len);
    try std.testing.expectEqual(@as(usize, 1), filled.relators[0].word.len);
}

test "connectivity obstruction reports disconnected output carrier for connected protocol carrier" {
    const in0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const in1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const p0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 0 } };
    const p1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .protocol = 1 } };
    const out0 = topology.Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const out1 = topology.Vertex{ .color = .{ .process = 1 }, .label = .{ .output = 1 } };
    const empty_vertices = [_]topology.Vertex{};
    const input_edge = [_]topology.Vertex{ in0, in1 };
    const input0 = [_]topology.Vertex{in0};
    const input1 = [_]topology.Vertex{in1};
    const protocol_edge = [_]topology.Vertex{ p0, p1 };
    const protocol0 = [_]topology.Vertex{p0};
    const protocol1 = [_]topology.Vertex{p1};
    const output0 = [_]topology.Vertex{out0};
    const output1 = [_]topology.Vertex{out1};
    const empty = topology.Simplex{ .vertices = &empty_vertices };
    const input = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &input_edge }} };
    const protocol = topology.Complex{ .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol_edge }} };
    const output = topology.Complex{ .simplexes = &[_]topology.Simplex{
        .{ .vertices = &output0 },
        .{ .vertices = &output1 },
    } };
    const execution_entries = [_]topology.SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{empty} } },
        .{ .simplex = .{ .vertices = &input0 }, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol0 }} } },
        .{ .simplex = .{ .vertices = &input1 }, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol1 }} } },
        .{ .simplex = .{ .vertices = &input_edge }, .image = .{ .parent = protocol, .simplexes = &[_]topology.Simplex{.{ .vertices = &protocol_edge }} } },
    };
    const task_entries = [_]topology.SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{empty} } },
        .{ .simplex = .{ .vertices = &input0 }, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{.{ .vertices = &output0 }} } },
        .{ .simplex = .{ .vertices = &input1 }, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{.{ .vertices = &output1 }} } },
        .{ .simplex = .{ .vertices = &input_edge }, .image = .{ .parent = output, .simplexes = &[_]topology.Simplex{
            .{ .vertices = &output0 },
            .{ .vertices = &output1 },
        } } },
    };

    var report = (try findConnectivityObstruction(std.testing.allocator, .{
        .domain = input,
        .codomain = protocol,
        .entries = &execution_entries,
    }, .{
        .domain = input,
        .codomain = output,
        .entries = &task_entries,
    })).?;
    defer report.deinit();

    try std.testing.expectEqual(ConnectivityObstructionKind.output_disconnected_for_connected_protocol, report.kind);
    try std.testing.expectEqual(@as(usize, 1), report.protocol_components);
    try std.testing.expectEqual(@as(usize, 2), report.output_components);
    try std.testing.expect(topology.Simplex.eql(report.inputSimplex(), .{ .vertices = &input_edge }));
}
