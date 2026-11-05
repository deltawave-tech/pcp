const std = @import("std");
const trace = @import("trace.zig");

pub const Color = union(enum) {
    hub: trace.NodeId,
    gateway: trace.NodeId,
    worker: trace.NodeId,
    api_caller: trace.NodeId,
    job: u32,
    session: u32,
    process: trace.NodeId,

    pub fn eql(a: Color, b: Color) bool {
        if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
        return switch (a) {
            .hub => |item| item == b.hub,
            .gateway => |item| item == b.gateway,
            .worker => |item| item == b.worker,
            .api_caller => |item| item == b.api_caller,
            .job => |item| item == b.job,
            .session => |item| item == b.session,
            .process => |item| item == b.process,
        };
    }

    pub fn order(self: Color) u8 {
        return switch (self) {
            .hub => 0,
            .gateway => 1,
            .worker => 2,
            .api_caller => 3,
            .job => 4,
            .session => 5,
            .process => 6,
        };
    }

    pub fn value(self: Color) u64 {
        return switch (self) {
            .hub => |item| item,
            .gateway => |item| item,
            .worker => |item| item,
            .api_caller => |item| item,
            .job => |item| item,
            .session => |item| item,
            .process => |item| item,
        };
    }

    pub fn lessThan(_: void, a: Color, b: Color) bool {
        if (a.order() != b.order()) return a.order() < b.order();
        return a.value() < b.value();
    }

    pub fn hash(self: Color) u64 {
        var hasher = std.hash.Wyhash.init(0x636f_6c6f_72);
        hasher.update(&.{self.order()});
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, self.value(), .little);
        hasher.update(&buf);
        return hasher.final();
    }
};

pub const Label = union(enum) {
    empty,
    input: i64,
    protocol: i64,
    output: i64,
    symbol: []const u8,

    pub fn eql(a: Label, b: Label) bool {
        if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
        return switch (a) {
            .empty => true,
            .input => |value| value == b.input,
            .protocol => |value| value == b.protocol,
            .output => |value| value == b.output,
            .symbol => |value| std.mem.eql(u8, value, b.symbol),
        };
    }

    pub fn order(self: Label) u8 {
        return switch (self) {
            .empty => 0,
            .input => 1,
            .protocol => 2,
            .output => 3,
            .symbol => 4,
        };
    }

    pub fn lessThan(_: void, a: Label, b: Label) bool {
        if (a.order() != b.order()) return a.order() < b.order();
        return switch (a) {
            .empty => false,
            .input => |value| value < b.input,
            .protocol => |value| value < b.protocol,
            .output => |value| value < b.output,
            .symbol => |value| std.mem.lessThan(u8, value, b.symbol),
        };
    }

    pub fn hash(self: Label) u64 {
        var hasher = std.hash.Wyhash.init(0x6c61_6265_6c);
        hasher.update(&.{self.order()});
        switch (self) {
            .empty => {},
            .input => |value| hashSigned(&hasher, value),
            .protocol => |value| hashSigned(&hasher, value),
            .output => |value| hashSigned(&hasher, value),
            .symbol => |value| hasher.update(value),
        }
        return hasher.final();
    }
};

pub const Vertex = struct {
    color: Color,
    label: Label,

    pub fn eql(a: Vertex, b: Vertex) bool {
        return Color.eql(a.color, b.color) and Label.eql(a.label, b.label);
    }

    pub fn lessThan(_: void, a: Vertex, b: Vertex) bool {
        if (!Color.eql(a.color, b.color)) return Color.lessThan({}, a.color, b.color);
        return Label.lessThan({}, a.label, b.label);
    }

    pub fn hash(self: Vertex) u64 {
        var hasher = std.hash.Wyhash.init(0x7665_7274_6578);
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, self.color.hash(), .little);
        hasher.update(&buf);
        std.mem.writeInt(u64, &buf, self.label.hash(), .little);
        hasher.update(&buf);
        return hasher.final();
    }
};

pub const OwnedVertexSet = struct {
    allocator: std.mem.Allocator,
    vertices: []Vertex,

    pub fn deinit(self: *OwnedVertexSet) void {
        self.allocator.free(self.vertices);
        self.vertices = &.{};
    }
};

pub const VertexId = usize;

pub const VertexRegistry = struct {
    allocator: std.mem.Allocator,
    vertices: std.ArrayList(Vertex),

    pub fn init(allocator: std.mem.Allocator) VertexRegistry {
        return .{
            .allocator = allocator,
            .vertices = std.ArrayList(Vertex).init(allocator),
        };
    }

    pub fn deinit(self: *VertexRegistry) void {
        self.vertices.deinit();
    }

    pub fn intern(self: *VertexRegistry, vertex: Vertex) !VertexId {
        if (self.find(vertex)) |index| return index;
        try self.vertices.append(vertex);
        std.mem.sort(Vertex, self.vertices.items, {}, Vertex.lessThan);
        return self.find(vertex).?;
    }

    pub fn find(self: VertexRegistry, vertex: Vertex) ?VertexId {
        for (self.vertices.items, 0..) |candidate, index| {
            if (Vertex.eql(candidate, vertex)) return index;
        }
        return null;
    }

    pub fn get(self: VertexRegistry, id: VertexId) ?Vertex {
        if (id >= self.vertices.items.len) return null;
        return self.vertices.items[id];
    }

    pub fn slice(self: VertexRegistry) []const Vertex {
        return self.vertices.items;
    }

    pub fn validateUniqueLabels(self: VertexRegistry) !void {
        for (self.vertices.items, 0..) |left, left_index| {
            for (self.vertices.items[left_index + 1 ..]) |right| {
                if (Vertex.eql(left, right)) return error.DuplicateInternedVertex;
            }
        }
    }
};

pub const OwnedSimplex = struct {
    allocator: std.mem.Allocator,
    vertices: []Vertex,

    pub fn simplex(self: *const OwnedSimplex) Simplex {
        return .{ .vertices = self.vertices };
    }

    pub fn deinit(self: *OwnedSimplex) void {
        self.allocator.free(self.vertices);
        self.vertices = &.{};
    }
};

pub const Simplex = struct {
    vertices: []const Vertex,

    pub fn validate(self: Simplex) !void {
        for (self.vertices, 0..) |left, left_index| {
            for (self.vertices[left_index + 1 ..]) |right| {
                if (Color.eql(left.color, right.color)) return error.DuplicateColor;
            }
        }
    }

    pub fn containsVertex(self: Simplex, vertex: Vertex) bool {
        for (self.vertices) |candidate| {
            if (Vertex.eql(candidate, vertex)) return true;
        }
        return false;
    }

    pub fn vertexWithColor(self: Simplex, color: Color) ?Vertex {
        for (self.vertices) |candidate| {
            if (Color.eql(candidate.color, color)) return candidate;
        }
        return null;
    }

    pub fn isSubsetOf(self: Simplex, other: Simplex) bool {
        for (self.vertices) |vertex| {
            if (!other.containsVertex(vertex)) return false;
        }
        return true;
    }

    pub fn eql(a: Simplex, b: Simplex) bool {
        return a.vertices.len == b.vertices.len and a.isSubsetOf(b);
    }

    pub fn isEmpty(self: Simplex) bool {
        return self.vertices.len == 0;
    }

    pub fn dimension(self: Simplex) isize {
        return @as(isize, @intCast(self.vertices.len)) - 1;
    }

    pub fn isProperSubsetOf(self: Simplex, other: Simplex) bool {
        return self.vertices.len < other.vertices.len and self.isSubsetOf(other);
    }

    pub fn disjointFrom(self: Simplex, other: Simplex) bool {
        for (self.vertices) |vertex| {
            if (other.containsVertex(vertex)) return false;
        }
        return true;
    }

    pub fn canonicalize(self: Simplex, allocator: std.mem.Allocator) !OwnedSimplex {
        try self.validate();
        const owned_vertices = try allocator.dupe(Vertex, self.vertices);
        std.mem.sort(Vertex, owned_vertices, {}, Vertex.lessThan);
        return .{ .allocator = allocator, .vertices = owned_vertices };
    }

    pub fn hash(self: Simplex, allocator: std.mem.Allocator) !u64 {
        var canonical = try self.canonicalize(allocator);
        defer canonical.deinit();

        var hasher = std.hash.Wyhash.init(0x7369_6d70_6c65_78);
        var buf: [8]u8 = undefined;
        for (canonical.vertices) |vertex| {
            std.mem.writeInt(u64, &buf, vertex.hash(), .little);
            hasher.update(&buf);
        }
        return hasher.final();
    }
};

pub const OwnedSimplexSet = struct {
    allocator: std.mem.Allocator,
    simplexes: []Simplex,

    pub fn containsSimplex(self: OwnedSimplexSet, simplex: Simplex) bool {
        for (self.simplexes) |candidate| {
            if (Simplex.eql(candidate, simplex)) return true;
        }
        return false;
    }

    pub fn deinit(self: *OwnedSimplexSet) void {
        for (self.simplexes) |simplex| {
            self.allocator.free(@constCast(simplex.vertices));
        }
        self.allocator.free(self.simplexes);
        self.simplexes = &.{};
    }
};

pub const OwnedComplex = struct {
    allocator: std.mem.Allocator,
    simplexes: []Simplex,

    pub fn complex(self: *const OwnedComplex) Complex {
        return .{ .simplexes = self.simplexes };
    }

    pub fn deinit(self: *OwnedComplex) void {
        for (self.simplexes) |simplex| {
            self.allocator.free(@constCast(simplex.vertices));
        }
        self.allocator.free(self.simplexes);
        self.simplexes = &.{};
    }
};

pub const Subcomplex = struct {
    parent: Complex,
    simplexes: []const Simplex,

    pub fn complex(self: Subcomplex) Complex {
        return .{ .simplexes = self.simplexes };
    }

    pub fn validate(self: Subcomplex) !void {
        try self.parent.validate();
        try self.complex().validate();
        for (self.simplexes) |simplex| {
            if (!self.parent.containsSimplex(simplex)) return error.SubcomplexOutsideParent;
        }
    }

    pub fn containsSimplex(self: Subcomplex, simplex: Simplex) bool {
        return self.parent.containsSimplex(simplex) and self.complex().containsSimplex(simplex);
    }

    pub fn closure(self: Subcomplex, allocator: std.mem.Allocator) !OwnedSubcomplex {
        var owned = try self.complex().faces(allocator);
        errdefer owned.deinit();
        return .{ .parent = self.parent, .owned = owned };
    }

    pub fn unionWith(self: Subcomplex, allocator: std.mem.Allocator, other: Subcomplex) !OwnedSubcomplex {
        try validateSameParent(self.parent, other.parent);
        var builder = SimplexCollectionBuilder.init(allocator);
        errdefer builder.deinit();
        for (self.simplexes) |simplex| try builder.add(simplex);
        for (other.simplexes) |simplex| try builder.add(simplex);
        var owned = try builder.toOwnedComplex();
        errdefer owned.deinit();
        return .{ .parent = self.parent, .owned = owned };
    }

    pub fn intersectWith(self: Subcomplex, allocator: std.mem.Allocator, other: Subcomplex) !OwnedSubcomplex {
        try validateSameParent(self.parent, other.parent);
        var self_faces = try self.complex().faces(allocator);
        defer self_faces.deinit();
        var other_faces = try other.complex().faces(allocator);
        defer other_faces.deinit();

        var builder = SimplexCollectionBuilder.init(allocator);
        errdefer builder.deinit();
        for (self_faces.simplexes) |simplex| {
            if (other_faces.complex().containsSimplex(simplex)) try builder.add(simplex);
        }

        var owned = try builder.toOwnedComplex();
        errdefer owned.deinit();
        return .{ .parent = self.parent, .owned = owned };
    }

    /// Returns the largest closed subcomplex contained in self and disjoint
    /// from other. Set-theoretic difference of complexes is not generally a
    /// complex; this keeps the output face-closed.
    pub fn differenceWith(self: Subcomplex, allocator: std.mem.Allocator, other: Subcomplex) !OwnedSubcomplex {
        try validateSameParent(self.parent, other.parent);
        var self_faces = try self.complex().faces(allocator);
        defer self_faces.deinit();

        var builder = SimplexCollectionBuilder.init(allocator);
        errdefer builder.deinit();
        for (self_faces.simplexes) |simplex| {
            if (!simplexHasFaceIn(simplex, other.complex())) try builder.add(simplex);
        }

        var owned = try builder.toOwnedComplex();
        errdefer owned.deinit();
        return .{ .parent = self.parent, .owned = owned };
    }

    pub fn image(
        self: Subcomplex,
        allocator: std.mem.Allocator,
        target_parent: Complex,
        context: anytype,
        mapVertex: anytype,
    ) !OwnedSubcomplex {
        var closure_value = try self.complex().faces(allocator);
        defer closure_value.deinit();

        var builder = SimplexCollectionBuilder.init(allocator);
        errdefer builder.deinit();
        for (closure_value.simplexes) |simplex| {
            const mapped_vertices = try allocator.alloc(Vertex, simplex.vertices.len);
            defer allocator.free(mapped_vertices);
            for (simplex.vertices, 0..) |vertex, index| {
                mapped_vertices[index] = mapVertex(context, vertex);
            }
            try builder.add(.{ .vertices = mapped_vertices });
        }

        var owned = try builder.toOwnedComplex();
        errdefer owned.deinit();
        for (owned.simplexes) |simplex| {
            if (!target_parent.containsSimplex(simplex)) return error.SubcomplexImageOutsideTarget;
        }
        return .{ .parent = target_parent, .owned = owned };
    }
};

pub const OwnedSubcomplex = struct {
    parent: Complex,
    owned: OwnedComplex,

    pub fn subcomplex(self: *const OwnedSubcomplex) Subcomplex {
        return .{ .parent = self.parent, .simplexes = self.owned.simplexes };
    }

    pub fn complex(self: *const OwnedSubcomplex) Complex {
        return self.owned.complex();
    }

    pub fn deinit(self: *OwnedSubcomplex) void {
        self.owned.deinit();
    }
};

pub const Complex = struct {
    /// Maximal or named simplexes. Faces are treated as members by closure.
    simplexes: []const Simplex,

    pub fn validate(self: Complex) !void {
        for (self.simplexes, 0..) |simplex, index| {
            try simplex.validate();
            for (self.simplexes[index + 1 ..]) |other| {
                if (Simplex.eql(simplex, other)) return error.DuplicateSimplex;
            }
        }
    }

    pub fn validateWithWitness(self: Complex) ?ValidationWitness {
        for (self.simplexes, 0..) |simplex, index| {
            simplex.validate() catch {
                return .{ .failure = .duplicate_color, .simplex_index = index };
            };
            for (self.simplexes[index + 1 ..], index + 1..) |other, other_index| {
                if (Simplex.eql(simplex, other)) {
                    return .{
                        .failure = .duplicate_simplex,
                        .simplex_index = index,
                        .other_simplex_index = other_index,
                    };
                }
            }
        }
        return null;
    }

    pub fn containsSimplex(self: Complex, simplex: Simplex) bool {
        return self.findContainingSimplexIndex(simplex) != null;
    }

    pub fn findSimplexIndex(self: Complex, simplex: Simplex) ?usize {
        for (self.simplexes, 0..) |candidate, index| {
            if (Simplex.eql(candidate, simplex)) return index;
        }
        return null;
    }

    pub fn findContainingSimplexIndex(self: Complex, simplex: Simplex) ?usize {
        for (self.simplexes, 0..) |candidate, index| {
            if (simplex.isSubsetOf(candidate)) return index;
        }
        return null;
    }

    pub fn dimension(self: Complex) isize {
        var result: isize = -1;
        for (self.simplexes) |simplex| {
            result = @max(result, simplex.dimension());
        }
        return result;
    }

    pub fn isPure(self: Complex) bool {
        var expected_dimension: ?isize = null;
        for (self.simplexes, 0..) |simplex, index| {
            if (!self.isMaximalIndex(index)) continue;
            const simplex_dimension = simplex.dimension();
            if (expected_dimension == null) {
                expected_dimension = simplex_dimension;
            } else if (expected_dimension.? != simplex_dimension) {
                return false;
            }
        }
        return true;
    }

    pub fn vertices(self: Complex, allocator: std.mem.Allocator) !OwnedVertexSet {
        var items = std.ArrayList(Vertex).init(allocator);
        errdefer items.deinit();

        for (self.simplexes) |simplex| {
            for (simplex.vertices) |vertex| {
                if (!containsVertex(items.items, vertex)) try items.append(vertex);
            }
        }
        std.mem.sort(Vertex, items.items, {}, Vertex.lessThan);

        return .{ .allocator = allocator, .vertices = try items.toOwnedSlice() };
    }

    pub fn faces(self: Complex, allocator: std.mem.Allocator) !OwnedComplex {
        var builder = SimplexCollectionBuilder.init(allocator);
        errdefer builder.deinit();
        for (self.simplexes) |simplex| {
            try addFacesOfSimplex(&builder, simplex);
        }
        return builder.toOwnedComplex();
    }

    pub fn skeleton(self: Complex, allocator: std.mem.Allocator, max_dimension: isize) !OwnedComplex {
        var faces_owned = try self.faces(allocator);
        defer faces_owned.deinit();

        var builder = SimplexCollectionBuilder.init(allocator);
        errdefer builder.deinit();
        for (faces_owned.simplexes) |simplex| {
            if (simplex.dimension() <= max_dimension) try builder.add(simplex);
        }
        return builder.toOwnedComplex();
    }

    pub fn boundary(self: Complex, allocator: std.mem.Allocator) !OwnedComplex {
        const top_dimension = self.dimension();
        if (top_dimension <= 0) {
            var builder = SimplexCollectionBuilder.init(allocator);
            return builder.toOwnedComplex();
        }

        var faces_owned = try self.faces(allocator);
        defer faces_owned.deinit();

        var builder = SimplexCollectionBuilder.init(allocator);
        errdefer builder.deinit();
        for (faces_owned.simplexes) |face| {
            if (face.dimension() != top_dimension - 1) continue;
            if (self.topFacetContainmentCount(face, top_dimension) == 1) {
                try builder.add(face);
            }
        }
        return builder.toOwnedComplex();
    }

    pub fn star(self: Complex, allocator: std.mem.Allocator, center: Simplex) !OwnedSimplexSet {
        var faces_owned = try self.faces(allocator);
        defer faces_owned.deinit();

        var builder = SimplexCollectionBuilder.init(allocator);
        errdefer builder.deinit();
        for (faces_owned.simplexes) |simplex| {
            if (center.isSubsetOf(simplex)) try builder.add(simplex);
        }
        return builder.toOwnedSet();
    }

    pub fn closedStar(self: Complex, allocator: std.mem.Allocator, center: Simplex) !OwnedComplex {
        var star_set = try self.star(allocator, center);
        defer star_set.deinit();

        var builder = SimplexCollectionBuilder.init(allocator);
        errdefer builder.deinit();
        for (star_set.simplexes) |simplex| {
            try addFacesOfSimplex(&builder, simplex);
        }
        return builder.toOwnedComplex();
    }

    pub fn link(self: Complex, allocator: std.mem.Allocator, center: Simplex) !OwnedComplex {
        var faces_owned = try self.faces(allocator);
        defer faces_owned.deinit();

        var builder = SimplexCollectionBuilder.init(allocator);
        errdefer builder.deinit();
        for (faces_owned.simplexes) |simplex| {
            if (!simplex.disjointFrom(center)) continue;
            var joined = try unionSimplexes(allocator, simplex, center);
            defer joined.deinit();
            if (self.containsSimplex(joined.simplex())) try builder.add(simplex);
        }
        return builder.toOwnedComplex();
    }

    pub fn asSubcomplex(self: Complex) Subcomplex {
        return .{ .parent = self, .simplexes = self.simplexes };
    }

    fn isMaximalIndex(self: Complex, index: usize) bool {
        const simplex = self.simplexes[index];
        for (self.simplexes, 0..) |other, other_index| {
            if (index == other_index) continue;
            if (simplex.isProperSubsetOf(other)) return false;
        }
        return true;
    }

    fn topFacetContainmentCount(self: Complex, face: Simplex, top_dimension: isize) usize {
        var count: usize = 0;
        for (self.simplexes) |simplex| {
            if (simplex.dimension() != top_dimension) continue;
            if (face.isSubsetOf(simplex)) count += 1;
        }
        return count;
    }
};

pub const ValidationFailure = enum {
    duplicate_color,
    duplicate_simplex,
};

pub const ValidationWitness = struct {
    failure: ValidationFailure,
    simplex_index: usize,
    other_simplex_index: ?usize = null,
};

pub fn joinComplexes(allocator: std.mem.Allocator, left: Complex, right: Complex) !OwnedComplex {
    try left.validate();
    try right.validate();

    var left_faces = try left.faces(allocator);
    defer left_faces.deinit();
    var right_faces = try right.faces(allocator);
    defer right_faces.deinit();

    var builder = SimplexCollectionBuilder.init(allocator);
    errdefer builder.deinit();
    for (left_faces.simplexes) |left_simplex| {
        for (right_faces.simplexes) |right_simplex| {
            var joined = try unionSimplexes(allocator, left_simplex, right_simplex);
            defer joined.deinit();
            try joined.simplex().validate();
            try builder.add(joined.simplex());
        }
    }
    return builder.toOwnedComplex();
}

pub const SimplicialMapChecks = struct {
    require_color_preserving: bool = false,
    require_rigid: bool = false,
};

pub const VertexMapEntry = struct {
    source: Vertex,
    target: Vertex,
};

pub const VertexMap = struct {
    domain: Complex,
    codomain: Complex,
    entries: []const VertexMapEntry,

    pub fn validate(self: VertexMap, allocator: std.mem.Allocator, checks: SimplicialMapChecks) !void {
        try self.domain.validate();
        try self.codomain.validate();
        try self.validateVertexCoverage(allocator, checks);
        try self.validateSimplicial(allocator, checks);
    }

    pub fn targetForVertex(self: VertexMap, source: Vertex) ?Vertex {
        for (self.entries) |entry| {
            if (Vertex.eql(entry.source, source)) return entry.target;
        }
        return null;
    }

    pub fn mapSimplex(self: VertexMap, allocator: std.mem.Allocator, simplex: Simplex) !OwnedSimplex {
        try simplex.validate();

        var vertices = std.ArrayList(Vertex).init(allocator);
        errdefer vertices.deinit();
        for (simplex.vertices) |source| {
            const target = self.targetForVertex(source) orelse return error.MissingVertexMapEntry;
            if (!containsVertex(vertices.items, target)) try vertices.append(target);
        }
        std.mem.sort(Vertex, vertices.items, {}, Vertex.lessThan);

        return .{ .allocator = allocator, .vertices = try vertices.toOwnedSlice() };
    }

    pub fn mapSubcomplex(self: VertexMap, allocator: std.mem.Allocator, subcomplex: Subcomplex) !OwnedSubcomplex {
        try validateComplexesSameByClosure(allocator, self.domain, subcomplex.parent);

        var faces_owned = try subcomplex.complex().faces(allocator);
        defer faces_owned.deinit();

        var builder = SimplexCollectionBuilder.init(allocator);
        errdefer builder.deinit();
        for (faces_owned.simplexes) |simplex| {
            var mapped = try self.mapSimplex(allocator, simplex);
            defer mapped.deinit();
            try builder.add(mapped.simplex());
        }

        var owned = try builder.toOwnedComplex();
        errdefer owned.deinit();
        for (owned.simplexes) |simplex| {
            if (!self.codomain.containsSimplex(simplex)) return error.NonSimplicialMap;
        }
        return .{ .parent = self.codomain, .owned = owned };
    }

    pub fn image(self: VertexMap, allocator: std.mem.Allocator) !OwnedSubcomplex {
        return self.mapSubcomplex(allocator, self.domain.asSubcomplex());
    }

    pub fn compose(self: VertexMap, allocator: std.mem.Allocator, next: VertexMap) !OwnedVertexMap {
        try validateComplexesSameByClosure(allocator, self.codomain, next.domain);

        var domain_vertices = try self.domain.vertices(allocator);
        defer domain_vertices.deinit();

        var entries = std.ArrayList(VertexMapEntry).init(allocator);
        errdefer entries.deinit();
        for (domain_vertices.vertices) |source| {
            const middle = self.targetForVertex(source) orelse return error.MissingVertexMapEntry;
            const target = next.targetForVertex(middle) orelse return error.MissingVertexMapEntry;
            try entries.append(.{ .source = source, .target = target });
        }

        return .{
            .allocator = allocator,
            .domain = self.domain,
            .codomain = next.codomain,
            .entries = try entries.toOwnedSlice(),
        };
    }

    fn validateVertexCoverage(self: VertexMap, allocator: std.mem.Allocator, checks: SimplicialMapChecks) !void {
        var domain_vertices = try self.domain.vertices(allocator);
        defer domain_vertices.deinit();
        var codomain_vertices = try self.codomain.vertices(allocator);
        defer codomain_vertices.deinit();

        var seen = try allocator.alloc(bool, domain_vertices.vertices.len);
        defer allocator.free(seen);
        @memset(seen, false);

        for (self.entries) |entry| {
            const source_index = indexOfVertex(domain_vertices.vertices, entry.source) orelse {
                return error.SourceVertexOutsideDomain;
            };
            if (seen[source_index]) return error.DuplicateVertexMapSource;
            seen[source_index] = true;

            if (!containsVertex(codomain_vertices.vertices, entry.target)) {
                return error.TargetVertexOutsideCodomain;
            }
            if (checks.require_color_preserving and !Color.eql(entry.source.color, entry.target.color)) {
                return error.NonColorPreservingMap;
            }
        }

        for (seen) |is_seen| {
            if (!is_seen) return error.MissingVertexMapEntry;
        }
    }

    fn validateSimplicial(self: VertexMap, allocator: std.mem.Allocator, checks: SimplicialMapChecks) !void {
        var faces_owned = try self.domain.faces(allocator);
        defer faces_owned.deinit();

        for (faces_owned.simplexes) |simplex| {
            var mapped = try self.mapSimplex(allocator, simplex);
            defer mapped.deinit();

            if (!self.codomain.containsSimplex(mapped.simplex())) return error.NonSimplicialMap;
            if (checks.require_rigid and mapped.vertices.len != simplex.vertices.len) {
                return error.NonRigidMap;
            }
        }
    }
};

pub const OwnedVertexMap = struct {
    allocator: std.mem.Allocator,
    domain: Complex,
    codomain: Complex,
    entries: []VertexMapEntry,

    pub fn vertexMap(self: *const OwnedVertexMap) VertexMap {
        return .{
            .domain = self.domain,
            .codomain = self.codomain,
            .entries = self.entries,
        };
    }

    pub fn deinit(self: *OwnedVertexMap) void {
        self.allocator.free(self.entries);
        self.entries = &.{};
    }
};

pub const CarrierChecks = struct {
    require_monotone: bool = true,
    require_strict: bool = false,
    require_rigid: bool = false,
    require_chromatic: bool = false,
};

pub const SimplicialCarrierEntry = struct {
    simplex: Simplex,
    image: Subcomplex,
};

pub const SimplicialCarrierMap = struct {
    domain: Complex,
    codomain: Complex,
    entries: []const SimplicialCarrierEntry,

    pub fn validate(self: SimplicialCarrierMap, allocator: std.mem.Allocator, checks: CarrierChecks) !void {
        try self.domain.validate();
        try self.codomain.validate();
        try self.validateEntries(allocator);

        if (checks.require_monotone) try self.validateMonotone(allocator);
        if (checks.require_strict) try self.validateStrict(allocator);
        if (checks.require_rigid or checks.require_chromatic) try self.validateRigid();
        if (checks.require_chromatic) try self.validateChromatic(allocator);
    }

    pub fn imageOf(self: SimplicialCarrierMap, simplex: Simplex) ?Subcomplex {
        for (self.entries) |entry| {
            if (Simplex.eql(entry.simplex, simplex)) return entry.image;
        }
        return null;
    }

    pub fn carries(self: SimplicialCarrierMap, input_simplex: Simplex, output_simplex: Simplex) bool {
        const image_value = self.imageOf(input_simplex) orelse return false;
        return image_value.containsSimplex(output_simplex);
    }

    pub fn image(self: SimplicialCarrierMap, allocator: std.mem.Allocator) !OwnedSubcomplex {
        var builder = SimplexCollectionBuilder.init(allocator);
        errdefer builder.deinit();

        for (self.entries) |entry| {
            var faces_owned = try entry.image.complex().faces(allocator);
            defer faces_owned.deinit();
            for (faces_owned.simplexes) |simplex| try builder.add(simplex);
        }

        var owned = try builder.toOwnedComplex();
        errdefer owned.deinit();
        return .{ .parent = self.codomain, .owned = owned };
    }

    pub fn restrictToDomain(self: SimplicialCarrierMap, allocator: std.mem.Allocator, subdomain: Subcomplex) !OwnedSimplicialCarrierMap {
        try validateComplexesSameByClosure(allocator, self.domain, subdomain.parent);

        var faces_owned = try subdomain.complex().faces(allocator);
        defer faces_owned.deinit();

        var builder = CarrierMapBuilder.init(allocator, subdomain.complex(), self.codomain);
        errdefer builder.deinit();
        for (faces_owned.simplexes) |simplex| {
            const image_value = self.imageOf(simplex) orelse return error.MissingCarrierSimplex;
            try builder.add(simplex, image_value);
        }
        return builder.toOwned();
    }

    pub fn composeWithMap(self: SimplicialCarrierMap, allocator: std.mem.Allocator, map: VertexMap) !OwnedSimplicialCarrierMap {
        try validateComplexesSameByClosure(allocator, self.codomain, map.domain);

        var builder = CarrierMapBuilder.init(allocator, self.domain, map.codomain);
        errdefer builder.deinit();
        for (self.entries) |entry| {
            var mapped = try map.mapSubcomplex(allocator, entry.image);
            defer mapped.deinit();
            try builder.add(entry.simplex, mapped.subcomplex());
        }
        return builder.toOwned();
    }

    pub fn composeWithCarrier(self: SimplicialCarrierMap, allocator: std.mem.Allocator, next: SimplicialCarrierMap) !OwnedSimplicialCarrierMap {
        try validateComplexesSameByClosure(allocator, self.codomain, next.domain);

        var builder = CarrierMapBuilder.init(allocator, self.domain, next.codomain);
        errdefer builder.deinit();
        for (self.entries) |entry| {
            var image_faces = try entry.image.complex().faces(allocator);
            defer image_faces.deinit();

            var union_builder = SimplexCollectionBuilder.init(allocator);
            defer union_builder.deinit();
            for (image_faces.simplexes) |middle_simplex| {
                const next_image = next.imageOf(middle_simplex) orelse return error.MissingCarrierSimplex;
                var next_faces = try next_image.complex().faces(allocator);
                defer next_faces.deinit();
                for (next_faces.simplexes) |output_simplex| try union_builder.add(output_simplex);
            }

            var composed_image = try union_builder.toOwnedComplex();
            defer composed_image.deinit();
            try builder.add(entry.simplex, .{
                .parent = next.codomain,
                .simplexes = composed_image.simplexes,
            });
        }
        return builder.toOwned();
    }

    fn validateEntries(self: SimplicialCarrierMap, allocator: std.mem.Allocator) !void {
        var domain_faces = try self.domain.faces(allocator);
        defer domain_faces.deinit();

        var seen = try allocator.alloc(bool, domain_faces.simplexes.len);
        defer allocator.free(seen);
        @memset(seen, false);

        for (self.entries) |entry| {
            const simplex_index = indexOfSimplex(domain_faces.simplexes, entry.simplex) orelse {
                return error.CarrierSimplexOutsideDomain;
            };
            if (seen[simplex_index]) return error.DuplicateCarrierSimplex;
            seen[simplex_index] = true;

            try validateComplexesSameByClosure(allocator, self.codomain, entry.image.parent);
            try entry.image.validate();
        }

        for (seen) |is_seen| {
            if (!is_seen) return error.MissingCarrierSimplex;
        }
    }

    fn validateMonotone(self: SimplicialCarrierMap, allocator: std.mem.Allocator) !void {
        var domain_faces = try self.domain.faces(allocator);
        defer domain_faces.deinit();

        for (domain_faces.simplexes) |left| {
            const left_image = self.imageOf(left).?;
            for (domain_faces.simplexes) |right| {
                if (!left.isSubsetOf(right)) continue;
                const right_image = self.imageOf(right).?;
                if (!(try subcomplexSubsetOf(allocator, left_image, right_image))) {
                    return error.NonMonotonicCarrier;
                }
            }
        }
    }

    fn validateStrict(self: SimplicialCarrierMap, allocator: std.mem.Allocator) !void {
        var domain_faces = try self.domain.faces(allocator);
        defer domain_faces.deinit();

        for (domain_faces.simplexes) |left| {
            const left_image = self.imageOf(left).?;
            for (domain_faces.simplexes) |right| {
                const right_image = self.imageOf(right).?;
                var intersection = try intersectSimplexes(allocator, left, right);
                defer intersection.deinit();

                const intersection_image = self.imageOf(intersection.simplex()) orelse return error.MissingCarrierSimplex;
                var image_intersection = try left_image.intersectWith(allocator, right_image);
                defer image_intersection.deinit();

                if (!(try subcomplexesEqualByClosure(allocator, intersection_image, image_intersection.subcomplex()))) {
                    return error.NonStrictCarrier;
                }
            }
        }
    }

    fn validateRigid(self: SimplicialCarrierMap) !void {
        for (self.entries) |entry| {
            if (!entry.image.complex().isPure()) return error.NonRigidCarrier;
            if (entry.image.complex().dimension() != entry.simplex.dimension()) {
                return error.NonRigidCarrier;
            }
        }
    }

    fn validateChromatic(self: SimplicialCarrierMap, allocator: std.mem.Allocator) !void {
        for (self.entries) |entry| {
            if (!(try simplexColorsEqualSubcomplex(allocator, entry.simplex, entry.image))) {
                return error.NonChromaticCarrier;
            }
        }
    }
};

pub const OwnedSimplicialCarrierMap = struct {
    allocator: std.mem.Allocator,
    domain: Complex,
    codomain: Complex,
    entries: []SimplicialCarrierEntry,

    pub fn carrierMap(self: *const OwnedSimplicialCarrierMap) SimplicialCarrierMap {
        return .{
            .domain = self.domain,
            .codomain = self.codomain,
            .entries = self.entries,
        };
    }

    pub fn deinit(self: *OwnedSimplicialCarrierMap) void {
        for (self.entries) |entry| freeCarrierEntry(self.allocator, entry);
        self.allocator.free(self.entries);
        self.entries = &.{};
    }
};

pub fn validateCarriedDecisionMap(
    allocator: std.mem.Allocator,
    execution: SimplicialCarrierMap,
    task: SimplicialCarrierMap,
    decision: VertexMap,
) !void {
    try execution.validate(allocator, .{});
    try task.validate(allocator, .{});
    try decision.validate(allocator, .{ .require_color_preserving = true });
    try validateComplexesSameByClosure(allocator, execution.domain, task.domain);
    try validateComplexesSameByClosure(allocator, execution.codomain, decision.domain);
    try validateComplexesSameByClosure(allocator, task.codomain, decision.codomain);

    var decided_execution = try execution.composeWithMap(allocator, decision);
    defer decided_execution.deinit();
    const decided_carrier = decided_execution.carrierMap();

    var domain_faces = try execution.domain.faces(allocator);
    defer domain_faces.deinit();
    for (domain_faces.simplexes) |input_simplex| {
        const decided_image = decided_carrier.imageOf(input_simplex) orelse return error.MissingCarrierSimplex;
        const task_image = task.imageOf(input_simplex) orelse return error.MissingCarrierSimplex;
        if (!(try subcomplexSubsetOf(allocator, decided_image, task_image))) {
            return error.DecisionOutsideCarrier;
        }
    }
}

const SimplexCollectionBuilder = struct {
    allocator: std.mem.Allocator,
    simplexes: std.ArrayList(Simplex),

    pub fn init(allocator: std.mem.Allocator) SimplexCollectionBuilder {
        return .{
            .allocator = allocator,
            .simplexes = std.ArrayList(Simplex).init(allocator),
        };
    }

    pub fn add(self: *SimplexCollectionBuilder, simplex: Simplex) !void {
        try simplex.validate();
        for (self.simplexes.items) |candidate| {
            if (Simplex.eql(candidate, simplex)) return;
        }

        var canonical = try simplex.canonicalize(self.allocator);
        errdefer canonical.deinit();
        try self.simplexes.append(canonical.simplex());
        canonical.vertices = &.{};
    }

    pub fn toOwnedComplex(self: *SimplexCollectionBuilder) !OwnedComplex {
        const owned = try self.simplexes.toOwnedSlice();
        self.simplexes = std.ArrayList(Simplex).init(self.allocator);
        return .{ .allocator = self.allocator, .simplexes = owned };
    }

    pub fn toOwnedSet(self: *SimplexCollectionBuilder) !OwnedSimplexSet {
        const owned = try self.simplexes.toOwnedSlice();
        self.simplexes = std.ArrayList(Simplex).init(self.allocator);
        return .{ .allocator = self.allocator, .simplexes = owned };
    }

    pub fn deinit(self: *SimplexCollectionBuilder) void {
        for (self.simplexes.items) |simplex| {
            self.allocator.free(@constCast(simplex.vertices));
        }
        self.simplexes.deinit();
    }
};

pub const ComplexBuilder = SimplexCollectionBuilder;

fn hashSigned(hasher: anytype, value: i64) void {
    var buf: [8]u8 = undefined;
    std.mem.writeInt(i64, &buf, value, .little);
    hasher.update(&buf);
}

fn containsVertex(vertices: []const Vertex, vertex: Vertex) bool {
    for (vertices) |candidate| {
        if (Vertex.eql(candidate, vertex)) return true;
    }
    return false;
}

fn addFacesOfSimplex(builder: *SimplexCollectionBuilder, simplex: Simplex) !void {
    try simplex.validate();
    if (simplex.vertices.len >= @bitSizeOf(usize)) return error.SimplexTooLargeForFaceEnumeration;

    const mask_count = @as(usize, 1) << @intCast(simplex.vertices.len);
    var mask: usize = 0;
    while (mask < mask_count) : (mask += 1) {
        const vertex_count = @popCount(mask);
        const face_vertices = try builder.allocator.alloc(Vertex, vertex_count);
        defer builder.allocator.free(face_vertices);

        var out_index: usize = 0;
        for (simplex.vertices, 0..) |vertex, vertex_index| {
            const bit = @as(usize, 1) << @intCast(vertex_index);
            if ((mask & bit) == 0) continue;
            face_vertices[out_index] = vertex;
            out_index += 1;
        }

        try builder.add(.{ .vertices = face_vertices });
    }
}

fn unionSimplexes(allocator: std.mem.Allocator, left: Simplex, right: Simplex) !OwnedSimplex {
    try left.validate();
    try right.validate();

    var vertices = std.ArrayList(Vertex).init(allocator);
    errdefer vertices.deinit();
    for (left.vertices) |vertex| {
        if (!containsVertex(vertices.items, vertex)) try vertices.append(vertex);
    }
    for (right.vertices) |vertex| {
        if (!containsVertex(vertices.items, vertex)) try vertices.append(vertex);
    }
    std.mem.sort(Vertex, vertices.items, {}, Vertex.lessThan);

    return .{ .allocator = allocator, .vertices = try vertices.toOwnedSlice() };
}

fn simplexHasFaceIn(simplex: Simplex, complex: Complex) bool {
    const vertex_count = simplex.vertices.len;
    if (vertex_count >= @bitSizeOf(usize)) return true;

    const mask_count = @as(usize, 1) << @intCast(vertex_count);
    var scratch: [@bitSizeOf(usize)]Vertex = undefined;
    var mask: usize = 1;
    while (mask < mask_count) : (mask += 1) {
        const face_vertex_count = @popCount(mask);
        var out_index: usize = 0;
        for (simplex.vertices, 0..) |vertex, vertex_index| {
            const bit = @as(usize, 1) << @intCast(vertex_index);
            if ((mask & bit) == 0) continue;
            scratch[out_index] = vertex;
            out_index += 1;
        }
        if (complex.containsSimplex(.{ .vertices = scratch[0..face_vertex_count] })) return true;
    }
    return false;
}

fn validateSameParent(left: Complex, right: Complex) !void {
    if (left.simplexes.len != right.simplexes.len) return error.ParentComplexMismatch;
    for (left.simplexes, 0..) |simplex, index| {
        if (!Simplex.eql(simplex, right.simplexes[index])) return error.ParentComplexMismatch;
    }
}

fn validateComplexesSameByClosure(allocator: std.mem.Allocator, left: Complex, right: Complex) !void {
    if (!(try complexesEqualByClosure(allocator, left, right))) return error.ComplexMismatch;
}

fn complexesEqualByClosure(allocator: std.mem.Allocator, left: Complex, right: Complex) !bool {
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

fn subcomplexSubsetOf(allocator: std.mem.Allocator, left: Subcomplex, right: Subcomplex) !bool {
    var left_faces = try left.complex().faces(allocator);
    defer left_faces.deinit();

    for (left_faces.simplexes) |simplex| {
        if (!right.complex().containsSimplex(simplex)) return false;
    }
    return true;
}

fn subcomplexesEqualByClosure(allocator: std.mem.Allocator, left: Subcomplex, right: Subcomplex) !bool {
    return (try subcomplexSubsetOf(allocator, left, right)) and
        (try subcomplexSubsetOf(allocator, right, left));
}

fn indexOfVertex(vertices: []const Vertex, needle: Vertex) ?usize {
    for (vertices, 0..) |candidate, index| {
        if (Vertex.eql(candidate, needle)) return index;
    }
    return null;
}

fn indexOfSimplex(simplexes: []const Simplex, needle: Simplex) ?usize {
    for (simplexes, 0..) |candidate, index| {
        if (Simplex.eql(candidate, needle)) return index;
    }
    return null;
}

fn intersectSimplexes(allocator: std.mem.Allocator, left: Simplex, right: Simplex) !OwnedSimplex {
    try left.validate();
    try right.validate();

    var vertices = std.ArrayList(Vertex).init(allocator);
    errdefer vertices.deinit();
    for (left.vertices) |vertex| {
        if (right.containsVertex(vertex)) try vertices.append(vertex);
    }
    std.mem.sort(Vertex, vertices.items, {}, Vertex.lessThan);

    return .{ .allocator = allocator, .vertices = try vertices.toOwnedSlice() };
}

fn simplexColorsEqualSubcomplex(allocator: std.mem.Allocator, simplex: Simplex, subcomplex: Subcomplex) !bool {
    var image_vertices = try subcomplex.complex().vertices(allocator);
    defer image_vertices.deinit();

    for (simplex.vertices) |source_vertex| {
        var found = false;
        for (image_vertices.vertices) |image_vertex| {
            if (Color.eql(source_vertex.color, image_vertex.color)) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }

    for (image_vertices.vertices) |image_vertex| {
        var found = false;
        for (simplex.vertices) |source_vertex| {
            if (Color.eql(source_vertex.color, image_vertex.color)) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }

    return true;
}

fn freeCarrierEntry(allocator: std.mem.Allocator, entry: SimplicialCarrierEntry) void {
    allocator.free(@constCast(entry.simplex.vertices));
    for (entry.image.simplexes) |simplex| {
        allocator.free(@constCast(simplex.vertices));
    }
    allocator.free(@constCast(entry.image.simplexes));
}

const CarrierMapBuilder = struct {
    allocator: std.mem.Allocator,
    domain: Complex,
    codomain: Complex,
    entries: std.ArrayList(SimplicialCarrierEntry),

    pub fn init(allocator: std.mem.Allocator, domain: Complex, codomain: Complex) CarrierMapBuilder {
        return .{
            .allocator = allocator,
            .domain = domain,
            .codomain = codomain,
            .entries = std.ArrayList(SimplicialCarrierEntry).init(allocator),
        };
    }

    pub fn add(self: *CarrierMapBuilder, simplex: Simplex, image: Subcomplex) !void {
        if (!self.domain.containsSimplex(simplex)) return error.CarrierSimplexOutsideDomain;
        try validateComplexesSameByClosure(self.allocator, self.codomain, image.parent);

        var owned_simplex = try simplex.canonicalize(self.allocator);
        errdefer owned_simplex.deinit();
        var closed_image = try image.closure(self.allocator);
        errdefer closed_image.deinit();

        try self.entries.append(.{
            .simplex = owned_simplex.simplex(),
            .image = .{
                .parent = self.codomain,
                .simplexes = closed_image.owned.simplexes,
            },
        });
        owned_simplex.vertices = &.{};
        closed_image.owned.simplexes = &.{};
    }

    pub fn toOwned(self: *CarrierMapBuilder) !OwnedSimplicialCarrierMap {
        const entries = try self.entries.toOwnedSlice();
        self.entries = std.ArrayList(SimplicialCarrierEntry).init(self.allocator);
        return .{
            .allocator = self.allocator,
            .domain = self.domain,
            .codomain = self.codomain,
            .entries = entries,
        };
    }

    pub fn deinit(self: *CarrierMapBuilder) void {
        for (self.entries.items) |entry| freeCarrierEntry(self.allocator, entry);
        self.entries.deinit();
    }
};

pub const SimplicialCarrierMapBuilder = CarrierMapBuilder;

pub const CarrierEdge = struct {
    input_index: usize,
    output_indices: []const usize,
};

pub const CarrierMap = struct {
    input: Complex,
    output: Complex,
    carries: []const CarrierEdge,

    pub fn validate(self: CarrierMap) !void {
        try self.input.validate();
        try self.output.validate();
        try self.validateEdges();
        try self.validateMonotonic();
    }

    pub fn outputCarriedByInput(self: CarrierMap, input_index: usize, output_index: usize) bool {
        const edge = self.edgeForInput(input_index) orelse return false;
        for (edge.output_indices) |candidate| {
            if (candidate == output_index) return true;
        }
        return false;
    }

    fn validateEdges(self: CarrierMap) !void {
        if (self.carries.len != self.input.simplexes.len) return error.MissingCarrierEdge;

        for (self.input.simplexes, 0..) |_, input_index| {
            var seen = false;
            for (self.carries) |edge| {
                if (edge.input_index != input_index) continue;
                if (seen) return error.DuplicateCarrierEdge;
                seen = true;
                for (edge.output_indices) |output_index| {
                    if (output_index >= self.output.simplexes.len) return error.OutputIndexOutOfRange;
                }
            }
            if (!seen) return error.MissingCarrierEdge;
        }
    }

    fn validateMonotonic(self: CarrierMap) !void {
        for (self.input.simplexes, 0..) |left, left_index| {
            const left_edge = self.edgeForInput(left_index).?;
            for (self.input.simplexes, 0..) |right, right_index| {
                if (left_index == right_index) continue;
                if (!left.isSubsetOf(right)) continue;

                const right_edge = self.edgeForInput(right_index).?;
                for (left_edge.output_indices) |output_index| {
                    if (!containsIndex(right_edge.output_indices, output_index)) {
                        return error.NonMonotonicCarrier;
                    }
                }
            }
        }
    }

    fn edgeForInput(self: CarrierMap, input_index: usize) ?CarrierEdge {
        for (self.carries) |edge| {
            if (edge.input_index == input_index) return edge;
        }
        return null;
    }
};

pub const ExecutionEdge = struct {
    input_index: usize,
    protocol_indices: []const usize,
};

pub const ExecutionCarrier = struct {
    input: Complex,
    protocol: Complex,
    carries: []const ExecutionEdge,

    pub fn validate(self: ExecutionCarrier) !void {
        try self.input.validate();
        try self.protocol.validate();
        try self.validateEdges();
        try self.validateMonotonic();
    }

    pub fn protocolCarriedByInput(self: ExecutionCarrier, input_index: usize, protocol_index: usize) bool {
        const edge = self.edgeForInput(input_index) orelse return false;
        return containsIndex(edge.protocol_indices, protocol_index);
    }

    fn validateEdges(self: ExecutionCarrier) !void {
        if (self.carries.len != self.input.simplexes.len) return error.MissingExecutionEdge;

        for (self.input.simplexes, 0..) |_, input_index| {
            var seen = false;
            for (self.carries) |edge| {
                if (edge.input_index != input_index) continue;
                if (seen) return error.DuplicateExecutionEdge;
                seen = true;
                if (edge.protocol_indices.len == 0) return error.EmptyExecutionCarrier;
                for (edge.protocol_indices) |protocol_index| {
                    if (protocol_index >= self.protocol.simplexes.len) return error.ProtocolIndexOutOfRange;
                }
            }
            if (!seen) return error.MissingExecutionEdge;
        }
    }

    fn validateMonotonic(self: ExecutionCarrier) !void {
        for (self.input.simplexes, 0..) |left, left_index| {
            const left_edge = self.edgeForInput(left_index).?;
            for (self.input.simplexes, 0..) |right, right_index| {
                if (left_index == right_index) continue;
                if (!left.isSubsetOf(right)) continue;

                const right_edge = self.edgeForInput(right_index).?;
                for (left_edge.protocol_indices) |protocol_index| {
                    if (!containsIndex(right_edge.protocol_indices, protocol_index)) {
                        return error.NonMonotonicExecutionCarrier;
                    }
                }
            }
        }
    }

    fn edgeForInput(self: ExecutionCarrier, input_index: usize) ?ExecutionEdge {
        for (self.carries) |edge| {
            if (edge.input_index == input_index) return edge;
        }
        return null;
    }
};

pub const DecisionEdge = struct {
    input_index: usize,
    protocol_index: usize,
    output_index: usize,
};

pub const ProtocolDecisionEdge = struct {
    protocol_index: usize,
    output_index: usize,
};

pub const ExecutionObservation = struct {
    input_index: usize,
    protocol_index: usize,
};

pub fn validateDecisionMap(
    allocator: std.mem.Allocator,
    protocol: Complex,
    carrier: CarrierMap,
    decisions: []const DecisionEdge,
) !void {
    try protocol.validate();
    try carrier.validate();

    if (decisions.len != protocol.simplexes.len) return error.MissingDecision;

    var seen = try allocator.alloc(bool, protocol.simplexes.len);
    defer allocator.free(seen);
    @memset(seen, false);

    for (decisions) |decision| {
        if (decision.input_index >= carrier.input.simplexes.len) return error.InputIndexOutOfRange;
        if (decision.protocol_index >= protocol.simplexes.len) return error.ProtocolIndexOutOfRange;
        if (decision.output_index >= carrier.output.simplexes.len) return error.OutputIndexOutOfRange;
        if (seen[decision.protocol_index]) return error.DuplicateDecision;
        seen[decision.protocol_index] = true;

        if (!carrier.outputCarriedByInput(decision.input_index, decision.output_index)) {
            return error.DecisionOutsideCarrier;
        }
    }

    for (seen) |is_seen| {
        if (!is_seen) return error.MissingDecision;
    }

    try validateDecisionCompatibility(protocol, carrier.output, decisions);
}

pub fn validateExecutionDecisionMap(
    allocator: std.mem.Allocator,
    execution: ExecutionCarrier,
    carrier: CarrierMap,
    decisions: []const ProtocolDecisionEdge,
) !void {
    try execution.validate();
    try carrier.validate();
    try validateIndexedComplexMatch(execution.input, carrier.input);

    if (decisions.len != execution.protocol.simplexes.len) return error.MissingDecision;

    var seen = try allocator.alloc(bool, execution.protocol.simplexes.len);
    defer allocator.free(seen);
    @memset(seen, false);

    for (decisions) |decision| {
        if (decision.protocol_index >= execution.protocol.simplexes.len) return error.ProtocolIndexOutOfRange;
        if (decision.output_index >= carrier.output.simplexes.len) return error.OutputIndexOutOfRange;
        if (seen[decision.protocol_index]) return error.DuplicateDecision;
        seen[decision.protocol_index] = true;

        var reachable = false;
        for (execution.carries) |edge| {
            if (!containsIndex(edge.protocol_indices, decision.protocol_index)) continue;
            reachable = true;
            if (!carrier.outputCarriedByInput(edge.input_index, decision.output_index)) {
                return error.DecisionOutsideCarrier;
            }
        }
        if (!reachable) return error.ProtocolOutsideExecutionCarrier;
    }

    for (seen) |is_seen| {
        if (!is_seen) return error.MissingDecision;
    }

    try validateProtocolDecisionCompatibility(execution.protocol, carrier.output, decisions);
}

pub fn validateExecutionObservations(
    allocator: std.mem.Allocator,
    execution: ExecutionCarrier,
    observations: []const ExecutionObservation,
) !void {
    try execution.validate();

    const slot_count = execution.input.simplexes.len * execution.protocol.simplexes.len;
    var seen = try allocator.alloc(bool, slot_count);
    defer allocator.free(seen);
    @memset(seen, false);

    for (observations) |observation| {
        if (observation.input_index >= execution.input.simplexes.len) return error.InputIndexOutOfRange;
        if (observation.protocol_index >= execution.protocol.simplexes.len) return error.ProtocolIndexOutOfRange;
        if (!execution.protocolCarriedByInput(observation.input_index, observation.protocol_index)) {
            return error.ProtocolOutsideExecutionCarrier;
        }
        seen[observation.input_index * execution.protocol.simplexes.len + observation.protocol_index] = true;
    }

    for (execution.carries) |edge| {
        for (edge.protocol_indices) |protocol_index| {
            const slot = edge.input_index * execution.protocol.simplexes.len + protocol_index;
            if (!seen[slot]) return error.MissingExecutionObservation;
        }
    }
}

fn containsIndex(indices: []const usize, needle: usize) bool {
    for (indices) |candidate| {
        if (candidate == needle) return true;
    }
    return false;
}

fn validateIndexedComplexMatch(left: Complex, right: Complex) !void {
    if (left.simplexes.len != right.simplexes.len) return error.ComplexMismatch;
    for (left.simplexes, 0..) |left_simplex, index| {
        if (!Simplex.eql(left_simplex, right.simplexes[index])) return error.ComplexMismatch;
    }
}

fn validateDecisionCompatibility(
    protocol: Complex,
    output: Complex,
    decisions: []const DecisionEdge,
) !void {
    for (decisions, 0..) |left, left_index| {
        const left_protocol = protocol.simplexes[left.protocol_index];
        const left_output = output.simplexes[left.output_index];

        for (decisions[left_index + 1 ..]) |right| {
            const right_protocol = protocol.simplexes[right.protocol_index];
            const right_output = output.simplexes[right.output_index];

            for (left_protocol.vertices) |shared_view| {
                if (!right_protocol.containsVertex(shared_view)) continue;

                const left_decision = left_output.vertexWithColor(shared_view.color) orelse {
                    return error.SharedViewMissingOutputVertex;
                };
                const right_decision = right_output.vertexWithColor(shared_view.color) orelse {
                    return error.SharedViewMissingOutputVertex;
                };
                if (!Vertex.eql(left_decision, right_decision)) {
                    return error.IncompatibleDecisionOnSharedView;
                }
            }
        }
    }
}

fn validateProtocolDecisionCompatibility(
    protocol: Complex,
    output: Complex,
    decisions: []const ProtocolDecisionEdge,
) !void {
    for (decisions, 0..) |left, left_index| {
        const left_protocol = protocol.simplexes[left.protocol_index];
        const left_output = output.simplexes[left.output_index];

        for (decisions[left_index + 1 ..]) |right| {
            const right_protocol = protocol.simplexes[right.protocol_index];
            const right_output = output.simplexes[right.output_index];

            for (left_protocol.vertices) |shared_view| {
                if (!right_protocol.containsVertex(shared_view)) continue;

                const left_decision = left_output.vertexWithColor(shared_view.color) orelse {
                    return error.SharedViewMissingOutputVertex;
                };
                const right_decision = right_output.vertexWithColor(shared_view.color) orelse {
                    return error.SharedViewMissingOutputVertex;
                };
                if (!Vertex.eql(left_decision, right_decision)) {
                    return error.IncompatibleDecisionOnSharedView;
                }
            }
        }
    }
}

test "simplex rejects duplicate colors" {
    const vertices = [_]Vertex{
        .{ .color = .{ .worker = 1 }, .label = .{ .input = 0 } },
        .{ .color = .{ .worker = 1 }, .label = .{ .input = 1 } },
    };

    try std.testing.expectError(error.DuplicateColor, (Simplex{ .vertices = &vertices }).validate());
}

test "simplex equality is order independent" {
    const a_vertices = [_]Vertex{
        .{ .color = .{ .gateway = 1 }, .label = .{ .symbol = "ready" } },
        .{ .color = .{ .worker = 2 }, .label = .{ .symbol = "joined" } },
    };
    const b_vertices = [_]Vertex{
        .{ .color = .{ .worker = 2 }, .label = .{ .symbol = "joined" } },
        .{ .color = .{ .gateway = 1 }, .label = .{ .symbol = "ready" } },
    };

    try std.testing.expect(Simplex.eql(.{ .vertices = &a_vertices }, .{ .vertices = &b_vertices }));
}

test "complex membership is closed under faces" {
    const facet_vertices = [_]Vertex{
        .{ .color = .{ .gateway = 1 }, .label = .{ .symbol = "ready" } },
        .{ .color = .{ .worker = 2 }, .label = .{ .symbol = "joined" } },
    };
    const face_vertices = [_]Vertex{
        .{ .color = .{ .worker = 2 }, .label = .{ .symbol = "joined" } },
    };
    const missing_vertices = [_]Vertex{
        .{ .color = .{ .worker = 3 }, .label = .{ .symbol = "joined" } },
    };
    const empty_vertices = [_]Vertex{};

    const complex = Complex{ .simplexes = &[_]Simplex{.{ .vertices = &facet_vertices }} };

    try std.testing.expect(complex.containsSimplex(.{ .vertices = &facet_vertices }));
    try std.testing.expect(complex.containsSimplex(.{ .vertices = &face_vertices }));
    try std.testing.expect(complex.containsSimplex(.{ .vertices = &empty_vertices }));
    try std.testing.expect(!complex.containsSimplex(.{ .vertices = &missing_vertices }));
    try std.testing.expectEqual(@as(?usize, null), complex.findSimplexIndex(.{ .vertices = &face_vertices }));
    try std.testing.expectEqual(@as(?usize, 0), complex.findContainingSimplexIndex(.{ .vertices = &face_vertices }));
}

test "complex rejects duplicate named simplexes" {
    const vertices = [_]Vertex{
        .{ .color = .{ .gateway = 1 }, .label = .{ .symbol = "ready" } },
    };
    const duplicate = [_]Simplex{
        .{ .vertices = &vertices },
        .{ .vertices = &vertices },
    };

    try std.testing.expectError(error.DuplicateSimplex, (Complex{ .simplexes = &duplicate }).validate());
}

test "simplex canonicalization and hashing are order independent" {
    const ordered = [_]Vertex{
        .{ .color = .{ .gateway = 1 }, .label = .{ .symbol = "ready" } },
        .{ .color = .{ .worker = 2 }, .label = .{ .protocol = 7 } },
    };
    const reversed = [_]Vertex{
        .{ .color = .{ .worker = 2 }, .label = .{ .protocol = 7 } },
        .{ .color = .{ .gateway = 1 }, .label = .{ .symbol = "ready" } },
    };

    var canonical = try (Simplex{ .vertices = &reversed }).canonicalize(std.testing.allocator);
    defer canonical.deinit();

    try std.testing.expect(Vertex.eql(ordered[0], canonical.vertices[0]));
    try std.testing.expect(Vertex.eql(ordered[1], canonical.vertices[1]));
    try std.testing.expectEqual(
        try (Simplex{ .vertices = &ordered }).hash(std.testing.allocator),
        try (Simplex{ .vertices = &reversed }).hash(std.testing.allocator),
    );
}

test "vertex registry interns color label pairs deterministically" {
    var registry = VertexRegistry.init(std.testing.allocator);
    defer registry.deinit();

    const gateway = Vertex{ .color = .{ .gateway = 1 }, .label = .{ .symbol = "ready" } };
    const worker = Vertex{ .color = .{ .worker = 2 }, .label = .{ .symbol = "ready" } };

    const gateway_id = try registry.intern(gateway);
    const worker_id = try registry.intern(worker);
    const duplicate_gateway_id = try registry.intern(gateway);

    try std.testing.expectEqual(gateway_id, duplicate_gateway_id);
    try std.testing.expect(registry.find(worker) == worker_id);
    try std.testing.expect(Vertex.eql(gateway, registry.get(gateway_id).?));
    try registry.validateUniqueLabels();
}

test "complex enumerates vertices faces dimension and purity" {
    const a = Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const triangle_vertices = [_]Vertex{ a, b, c };
    const triangle = Simplex{ .vertices = &triangle_vertices };
    const complex = Complex{ .simplexes = &[_]Simplex{triangle} };

    var vertices = try complex.vertices(std.testing.allocator);
    defer vertices.deinit();
    var faces = try complex.faces(std.testing.allocator);
    defer faces.deinit();

    try std.testing.expectEqual(@as(usize, 3), vertices.vertices.len);
    try std.testing.expectEqual(@as(usize, 8), faces.simplexes.len);
    try std.testing.expectEqual(@as(isize, 2), complex.dimension());
    try std.testing.expect(complex.isPure());
    try std.testing.expect(faces.complex().containsSimplex(.{ .vertices = &[_]Vertex{ a, b } }));
    try std.testing.expect(faces.complex().containsSimplex(.{ .vertices = &[_]Vertex{} }));
}

test "complex purity uses maximal simplexes" {
    const a = Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const d = Vertex{ .color = .{ .process = 3 }, .label = .{ .input = 3 } };
    const edge_ab_vertices = [_]Vertex{ a, b };
    const edge_ac_vertices = [_]Vertex{ a, c };
    const vertex_d = [_]Vertex{d};

    try std.testing.expect((Complex{ .simplexes = &[_]Simplex{
        .{ .vertices = &edge_ab_vertices },
        .{ .vertices = &edge_ac_vertices },
    } }).isPure());
    try std.testing.expect(!(Complex{ .simplexes = &[_]Simplex{
        .{ .vertices = &edge_ab_vertices },
        .{ .vertices = &vertex_d },
    } }).isPure());
}

test "skeleton and boundary are explicit finite complexes" {
    const a = Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const d = Vertex{ .color = .{ .process = 3 }, .label = .{ .input = 3 } };
    const abc_vertices = [_]Vertex{ a, b, c };
    const abd_vertices = [_]Vertex{ a, b, d };
    const ab_vertices = [_]Vertex{ a, b };
    const ac_vertices = [_]Vertex{ a, c };
    const bc_vertices = [_]Vertex{ b, c };
    const ad_vertices = [_]Vertex{ a, d };
    const bd_vertices = [_]Vertex{ b, d };
    const complex = Complex{ .simplexes = &[_]Simplex{
        .{ .vertices = &abc_vertices },
        .{ .vertices = &abd_vertices },
    } };

    var skeleton = try complex.skeleton(std.testing.allocator, 1);
    defer skeleton.deinit();
    var boundary = try complex.boundary(std.testing.allocator);
    defer boundary.deinit();

    try std.testing.expect(!skeleton.complex().containsSimplex(.{ .vertices = &abc_vertices }));
    try std.testing.expect(skeleton.complex().containsSimplex(.{ .vertices = &ab_vertices }));
    try std.testing.expect(!boundary.complex().containsSimplex(.{ .vertices = &ab_vertices }));
    try std.testing.expect(boundary.complex().containsSimplex(.{ .vertices = &ac_vertices }));
    try std.testing.expect(boundary.complex().containsSimplex(.{ .vertices = &bc_vertices }));
    try std.testing.expect(boundary.complex().containsSimplex(.{ .vertices = &ad_vertices }));
    try std.testing.expect(boundary.complex().containsSimplex(.{ .vertices = &bd_vertices }));
}

test "star closed star and link follow face closure" {
    const a = Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const triangle_vertices = [_]Vertex{ a, b, c };
    const center_vertices = [_]Vertex{a};
    const bc_vertices = [_]Vertex{ b, c };
    const triangle = Simplex{ .vertices = &triangle_vertices };
    const center = Simplex{ .vertices = &center_vertices };
    const complex = Complex{ .simplexes = &[_]Simplex{triangle} };

    var star = try complex.star(std.testing.allocator, center);
    defer star.deinit();
    var closed_star = try complex.closedStar(std.testing.allocator, center);
    defer closed_star.deinit();
    var link = try complex.link(std.testing.allocator, center);
    defer link.deinit();

    try std.testing.expect(star.containsSimplex(center));
    try std.testing.expect(star.containsSimplex(triangle));
    try std.testing.expect(!star.containsSimplex(.{ .vertices = &bc_vertices }));
    try std.testing.expect(closed_star.complex().containsSimplex(.{ .vertices = &bc_vertices }));
    try std.testing.expect(link.complex().containsSimplex(.{ .vertices = &bc_vertices }));
    try std.testing.expect(!link.complex().containsSimplex(center));
}

test "subcomplex operations keep outputs closed inside the parent" {
    const a = Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const c = Vertex{ .color = .{ .process = 2 }, .label = .{ .input = 2 } };
    const triangle_vertices = [_]Vertex{ a, b, c };
    const ab_vertices = [_]Vertex{ a, b };
    const ac_vertices = [_]Vertex{ a, c };
    const a_vertices = [_]Vertex{a};
    const b_vertices = [_]Vertex{b};
    const parent = Complex{ .simplexes = &[_]Simplex{.{ .vertices = &triangle_vertices }} };
    const left = Subcomplex{ .parent = parent, .simplexes = &[_]Simplex{.{ .vertices = &ab_vertices }} };
    const right = Subcomplex{ .parent = parent, .simplexes = &[_]Simplex{.{ .vertices = &ac_vertices }} };

    var union_value = try left.unionWith(std.testing.allocator, right);
    defer union_value.deinit();
    var intersection = try left.intersectWith(std.testing.allocator, right);
    defer intersection.deinit();
    var difference = try left.differenceWith(std.testing.allocator, right);
    defer difference.deinit();

    try union_value.subcomplex().validate();
    try intersection.subcomplex().validate();
    try difference.subcomplex().validate();
    try std.testing.expect(union_value.complex().containsSimplex(.{ .vertices = &ab_vertices }));
    try std.testing.expect(union_value.complex().containsSimplex(.{ .vertices = &ac_vertices }));
    try std.testing.expect(intersection.complex().containsSimplex(.{ .vertices = &a_vertices }));
    try std.testing.expect(!intersection.complex().containsSimplex(.{ .vertices = &b_vertices }));
    try std.testing.expect(difference.complex().containsSimplex(.{ .vertices = &b_vertices }));
    try std.testing.expect(!difference.complex().containsSimplex(.{ .vertices = &ab_vertices }));
}

test "subcomplex image maps closed faces into a target parent" {
    const a = Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const out_a = Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const out_b = Vertex{ .color = .{ .process = 1 }, .label = .{ .output = 1 } };
    const input_edge = [_]Vertex{ a, b };
    const output_edge = [_]Vertex{ out_a, out_b };
    const source_parent = Complex{ .simplexes = &[_]Simplex{.{ .vertices = &input_edge }} };
    const target_parent = Complex{ .simplexes = &[_]Simplex{.{ .vertices = &output_edge }} };
    const source = source_parent.asSubcomplex();
    const Mapper = struct {
        fn map(_: void, vertex: Vertex) Vertex {
            if (Color.eql(vertex.color, .{ .process = 0 })) {
                return .{ .color = vertex.color, .label = .{ .output = 0 } };
            }
            return .{ .color = vertex.color, .label = .{ .output = 1 } };
        }
    };

    var image = try source.image(std.testing.allocator, target_parent, {}, Mapper.map);
    defer image.deinit();

    try image.subcomplex().validate();
    try std.testing.expect(image.complex().containsSimplex(.{ .vertices = &output_edge }));
}

test "join builds the face-closed product over disjoint colors" {
    const a = Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const left_vertices = [_]Vertex{a};
    const right_vertices = [_]Vertex{b};
    const edge_vertices = [_]Vertex{ a, b };

    var joined = try joinComplexes(
        std.testing.allocator,
        .{ .simplexes = &[_]Simplex{.{ .vertices = &left_vertices }} },
        .{ .simplexes = &[_]Simplex{.{ .vertices = &right_vertices }} },
    );
    defer joined.deinit();

    try std.testing.expectEqual(@as(usize, 4), joined.simplexes.len);
    try std.testing.expect(joined.complex().containsSimplex(.{ .vertices = &edge_vertices }));
}

test "vertex maps validate simplicial color-preserving rigid images" {
    const a = Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 0 } };
    const b = Vertex{ .color = .{ .process = 1 }, .label = .{ .protocol = 1 } };
    const out_a = Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const out_b = Vertex{ .color = .{ .process = 1 }, .label = .{ .output = 1 } };
    const input_edge = [_]Vertex{ a, b };
    const output_edge = [_]Vertex{ out_a, out_b };
    const domain = Complex{ .simplexes = &[_]Simplex{.{ .vertices = &input_edge }} };
    const codomain = Complex{ .simplexes = &[_]Simplex{.{ .vertices = &output_edge }} };
    const entries = [_]VertexMapEntry{
        .{ .source = a, .target = out_a },
        .{ .source = b, .target = out_b },
    };
    const map = VertexMap{ .domain = domain, .codomain = codomain, .entries = &entries };

    try map.validate(std.testing.allocator, .{
        .require_color_preserving = true,
        .require_rigid = true,
    });

    var image = try map.image(std.testing.allocator);
    defer image.deinit();
    try std.testing.expect(image.complex().containsSimplex(.{ .vertices = &output_edge }));
}

test "vertex maps reject non-simplicial images and rigid collapses" {
    const a = Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 0 } };
    const b = Vertex{ .color = .{ .process = 1 }, .label = .{ .protocol = 1 } };
    const out_a = Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const out_b = Vertex{ .color = .{ .process = 1 }, .label = .{ .output = 1 } };
    const input_edge = [_]Vertex{ a, b };
    const output_a = [_]Vertex{out_a};
    const output_b = [_]Vertex{out_b};
    const output_edge = [_]Vertex{ out_a, out_b };
    const domain = Complex{ .simplexes = &[_]Simplex{.{ .vertices = &input_edge }} };

    const nonsimplicial_entries = [_]VertexMapEntry{
        .{ .source = a, .target = out_a },
        .{ .source = b, .target = out_b },
    };
    try std.testing.expectError(error.NonSimplicialMap, (VertexMap{
        .domain = domain,
        .codomain = .{ .simplexes = &[_]Simplex{
            .{ .vertices = &output_a },
            .{ .vertices = &output_b },
        } },
        .entries = &nonsimplicial_entries,
    }).validate(std.testing.allocator, .{}));

    const collapse_entries = [_]VertexMapEntry{
        .{ .source = a, .target = out_a },
        .{ .source = b, .target = out_a },
    };
    try std.testing.expectError(error.NonRigidMap, (VertexMap{
        .domain = domain,
        .codomain = .{ .simplexes = &[_]Simplex{.{ .vertices = &output_edge }} },
        .entries = &collapse_entries,
    }).validate(std.testing.allocator, .{ .require_rigid = true }));
}

test "simplicial carrier maps validate strict rigid chromatic carriers" {
    const a = Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const b = Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const out_a = Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const out_b = Vertex{ .color = .{ .process = 1 }, .label = .{ .output = 1 } };
    const empty_vertices = [_]Vertex{};
    const input_a = [_]Vertex{a};
    const input_b = [_]Vertex{b};
    const input_edge = [_]Vertex{ a, b };
    const output_a = [_]Vertex{out_a};
    const output_b = [_]Vertex{out_b};
    const output_edge = [_]Vertex{ out_a, out_b };
    const empty = Simplex{ .vertices = &empty_vertices };
    const domain = Complex{ .simplexes = &[_]Simplex{.{ .vertices = &input_edge }} };
    const codomain = Complex{ .simplexes = &[_]Simplex{.{ .vertices = &output_edge }} };
    const carrier_entries = [_]SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = codomain, .simplexes = &[_]Simplex{empty} } },
        .{
            .simplex = .{ .vertices = &input_a },
            .image = .{ .parent = codomain, .simplexes = &[_]Simplex{.{ .vertices = &output_a }} },
        },
        .{
            .simplex = .{ .vertices = &input_b },
            .image = .{ .parent = codomain, .simplexes = &[_]Simplex{.{ .vertices = &output_b }} },
        },
        .{
            .simplex = .{ .vertices = &input_edge },
            .image = .{ .parent = codomain, .simplexes = &[_]Simplex{.{ .vertices = &output_edge }} },
        },
    };
    const carrier = SimplicialCarrierMap{
        .domain = domain,
        .codomain = codomain,
        .entries = &carrier_entries,
    };

    try carrier.validate(std.testing.allocator, .{
        .require_strict = true,
        .require_rigid = true,
        .require_chromatic = true,
    });

    var image = try carrier.image(std.testing.allocator);
    defer image.deinit();
    var restricted = try carrier.restrictToDomain(std.testing.allocator, .{
        .parent = domain,
        .simplexes = &[_]Simplex{.{ .vertices = &input_a }},
    });
    defer restricted.deinit();

    try std.testing.expect(image.complex().containsSimplex(.{ .vertices = &output_edge }));
    try restricted.carrierMap().validate(std.testing.allocator, .{ .require_chromatic = true });
}

test "carrier calculus composes carrier maps and carried decision maps" {
    const in_a = Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const in_b = Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const view_a = Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 10 } };
    const view_b = Vertex{ .color = .{ .process = 1 }, .label = .{ .protocol = 20 } };
    const out_a = Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const out_b = Vertex{ .color = .{ .process = 1 }, .label = .{ .output = 1 } };
    const empty_vertices = [_]Vertex{};
    const input_a = [_]Vertex{in_a};
    const input_b = [_]Vertex{in_b};
    const input_edge = [_]Vertex{ in_a, in_b };
    const protocol_a = [_]Vertex{view_a};
    const protocol_b = [_]Vertex{view_b};
    const protocol_edge = [_]Vertex{ view_a, view_b };
    const output_a = [_]Vertex{out_a};
    const output_b = [_]Vertex{out_b};
    const output_edge = [_]Vertex{ out_a, out_b };
    const empty = Simplex{ .vertices = &empty_vertices };
    const input = Complex{ .simplexes = &[_]Simplex{.{ .vertices = &input_edge }} };
    const protocol = Complex{ .simplexes = &[_]Simplex{.{ .vertices = &protocol_edge }} };
    const output = Complex{ .simplexes = &[_]Simplex{.{ .vertices = &output_edge }} };

    const execution_entries = [_]SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = protocol, .simplexes = &[_]Simplex{empty} } },
        .{
            .simplex = .{ .vertices = &input_a },
            .image = .{ .parent = protocol, .simplexes = &[_]Simplex{.{ .vertices = &protocol_a }} },
        },
        .{
            .simplex = .{ .vertices = &input_b },
            .image = .{ .parent = protocol, .simplexes = &[_]Simplex{.{ .vertices = &protocol_b }} },
        },
        .{
            .simplex = .{ .vertices = &input_edge },
            .image = .{ .parent = protocol, .simplexes = &[_]Simplex{.{ .vertices = &protocol_edge }} },
        },
    };
    const protocol_to_output_entries = [_]SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = output, .simplexes = &[_]Simplex{empty} } },
        .{
            .simplex = .{ .vertices = &protocol_a },
            .image = .{ .parent = output, .simplexes = &[_]Simplex{.{ .vertices = &output_a }} },
        },
        .{
            .simplex = .{ .vertices = &protocol_b },
            .image = .{ .parent = output, .simplexes = &[_]Simplex{.{ .vertices = &output_b }} },
        },
        .{
            .simplex = .{ .vertices = &protocol_edge },
            .image = .{ .parent = output, .simplexes = &[_]Simplex{.{ .vertices = &output_edge }} },
        },
    };
    const task_entries = [_]SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = output, .simplexes = &[_]Simplex{empty} } },
        .{
            .simplex = .{ .vertices = &input_a },
            .image = .{ .parent = output, .simplexes = &[_]Simplex{.{ .vertices = &output_a }} },
        },
        .{
            .simplex = .{ .vertices = &input_b },
            .image = .{ .parent = output, .simplexes = &[_]Simplex{.{ .vertices = &output_b }} },
        },
        .{
            .simplex = .{ .vertices = &input_edge },
            .image = .{ .parent = output, .simplexes = &[_]Simplex{.{ .vertices = &output_edge }} },
        },
    };
    const execution = SimplicialCarrierMap{ .domain = input, .codomain = protocol, .entries = &execution_entries };
    const protocol_output = SimplicialCarrierMap{ .domain = protocol, .codomain = output, .entries = &protocol_to_output_entries };
    const task = SimplicialCarrierMap{ .domain = input, .codomain = output, .entries = &task_entries };
    const decision_entries = [_]VertexMapEntry{
        .{ .source = view_a, .target = out_a },
        .{ .source = view_b, .target = out_b },
    };
    const decision = VertexMap{ .domain = protocol, .codomain = output, .entries = &decision_entries };

    var composed = try execution.composeWithCarrier(std.testing.allocator, protocol_output);
    defer composed.deinit();

    try composed.carrierMap().validate(std.testing.allocator, .{ .require_chromatic = true });
    try std.testing.expect(composed.carrierMap().carries(.{ .vertices = &input_edge }, .{ .vertices = &output_edge }));
    try validateCarriedDecisionMap(std.testing.allocator, execution, task, decision);
}

test "carried decision maps reject task carrier escapes" {
    const in_a = Vertex{ .color = .{ .process = 0 }, .label = .{ .input = 0 } };
    const in_b = Vertex{ .color = .{ .process = 1 }, .label = .{ .input = 1 } };
    const view_a = Vertex{ .color = .{ .process = 0 }, .label = .{ .protocol = 10 } };
    const view_b = Vertex{ .color = .{ .process = 1 }, .label = .{ .protocol = 20 } };
    const out_a = Vertex{ .color = .{ .process = 0 }, .label = .{ .output = 0 } };
    const out_b = Vertex{ .color = .{ .process = 1 }, .label = .{ .output = 1 } };
    const out_b_bad = Vertex{ .color = .{ .process = 1 }, .label = .{ .output = 9 } };
    const empty_vertices = [_]Vertex{};
    const input_a = [_]Vertex{in_a};
    const input_b = [_]Vertex{in_b};
    const input_edge = [_]Vertex{ in_a, in_b };
    const protocol_a = [_]Vertex{view_a};
    const protocol_b = [_]Vertex{view_b};
    const protocol_edge = [_]Vertex{ view_a, view_b };
    const output_a = [_]Vertex{out_a};
    const output_b = [_]Vertex{out_b};
    const output_good_edge = [_]Vertex{ out_a, out_b };
    const output_bad_edge = [_]Vertex{ out_a, out_b_bad };
    const empty = Simplex{ .vertices = &empty_vertices };
    const input = Complex{ .simplexes = &[_]Simplex{.{ .vertices = &input_edge }} };
    const protocol = Complex{ .simplexes = &[_]Simplex{.{ .vertices = &protocol_edge }} };
    const output = Complex{ .simplexes = &[_]Simplex{
        .{ .vertices = &output_good_edge },
        .{ .vertices = &output_bad_edge },
    } };
    const execution_entries = [_]SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = protocol, .simplexes = &[_]Simplex{empty} } },
        .{
            .simplex = .{ .vertices = &input_a },
            .image = .{ .parent = protocol, .simplexes = &[_]Simplex{.{ .vertices = &protocol_a }} },
        },
        .{
            .simplex = .{ .vertices = &input_b },
            .image = .{ .parent = protocol, .simplexes = &[_]Simplex{.{ .vertices = &protocol_b }} },
        },
        .{
            .simplex = .{ .vertices = &input_edge },
            .image = .{ .parent = protocol, .simplexes = &[_]Simplex{.{ .vertices = &protocol_edge }} },
        },
    };
    const task_entries = [_]SimplicialCarrierEntry{
        .{ .simplex = empty, .image = .{ .parent = output, .simplexes = &[_]Simplex{empty} } },
        .{
            .simplex = .{ .vertices = &input_a },
            .image = .{ .parent = output, .simplexes = &[_]Simplex{.{ .vertices = &output_a }} },
        },
        .{
            .simplex = .{ .vertices = &input_b },
            .image = .{ .parent = output, .simplexes = &[_]Simplex{.{ .vertices = &output_b }} },
        },
        .{
            .simplex = .{ .vertices = &input_edge },
            .image = .{ .parent = output, .simplexes = &[_]Simplex{.{ .vertices = &output_good_edge }} },
        },
    };
    const bad_decision_entries = [_]VertexMapEntry{
        .{ .source = view_a, .target = out_a },
        .{ .source = view_b, .target = out_b_bad },
    };

    try std.testing.expectError(error.DecisionOutsideCarrier, validateCarriedDecisionMap(
        std.testing.allocator,
        .{ .domain = input, .codomain = protocol, .entries = &execution_entries },
        .{ .domain = input, .codomain = output, .entries = &task_entries },
        .{ .domain = protocol, .codomain = output, .entries = &bad_decision_entries },
    ));
}

test "carrier map validates monotonic output sets" {
    const input_vertices_a = [_]Vertex{
        .{ .color = .{ .worker = 0 }, .label = .{ .input = 0 } },
    };
    const input_vertices_ab = [_]Vertex{
        .{ .color = .{ .worker = 0 }, .label = .{ .input = 0 } },
        .{ .color = .{ .worker = 1 }, .label = .{ .input = 1 } },
    };
    const output_vertices_a = [_]Vertex{
        .{ .color = .{ .worker = 0 }, .label = .{ .output = 0 } },
    };
    const output_vertices_ab = [_]Vertex{
        .{ .color = .{ .worker = 0 }, .label = .{ .output = 0 } },
        .{ .color = .{ .worker = 1 }, .label = .{ .output = 1 } },
    };

    const input_simplexes = [_]Simplex{
        .{ .vertices = &input_vertices_a },
        .{ .vertices = &input_vertices_ab },
    };
    const output_simplexes = [_]Simplex{
        .{ .vertices = &output_vertices_a },
        .{ .vertices = &output_vertices_ab },
    };
    const edge_a_outputs = [_]usize{0};
    const edge_ab_outputs = [_]usize{ 0, 1 };
    const edges = [_]CarrierEdge{
        .{ .input_index = 0, .output_indices = &edge_a_outputs },
        .{ .input_index = 1, .output_indices = &edge_ab_outputs },
    };

    try (CarrierMap{
        .input = .{ .simplexes = &input_simplexes },
        .output = .{ .simplexes = &output_simplexes },
        .carries = &edges,
    }).validate();
}

test "carrier map rejects non-monotonic output sets" {
    const input_vertices_a = [_]Vertex{
        .{ .color = .{ .worker = 0 }, .label = .{ .input = 0 } },
    };
    const input_vertices_ab = [_]Vertex{
        .{ .color = .{ .worker = 0 }, .label = .{ .input = 0 } },
        .{ .color = .{ .worker = 1 }, .label = .{ .input = 1 } },
    };
    const output_vertices_a = [_]Vertex{
        .{ .color = .{ .worker = 0 }, .label = .{ .output = 0 } },
    };
    const output_vertices_ab = [_]Vertex{
        .{ .color = .{ .worker = 0 }, .label = .{ .output = 0 } },
        .{ .color = .{ .worker = 1 }, .label = .{ .output = 1 } },
    };

    const input_simplexes = [_]Simplex{
        .{ .vertices = &input_vertices_a },
        .{ .vertices = &input_vertices_ab },
    };
    const output_simplexes = [_]Simplex{
        .{ .vertices = &output_vertices_a },
        .{ .vertices = &output_vertices_ab },
    };
    const edge_a_outputs = [_]usize{ 0, 1 };
    const edge_ab_outputs = [_]usize{0};
    const edges = [_]CarrierEdge{
        .{ .input_index = 0, .output_indices = &edge_a_outputs },
        .{ .input_index = 1, .output_indices = &edge_ab_outputs },
    };

    try std.testing.expectError(error.NonMonotonicCarrier, (CarrierMap{
        .input = .{ .simplexes = &input_simplexes },
        .output = .{ .simplexes = &output_simplexes },
        .carries = &edges,
    }).validate());
}

test "execution carrier validates reachable protocol decisions through task carrier" {
    const input_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .input = 1 } },
    };
    const protocol_a_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .protocol = 10 } },
    };
    const protocol_b_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .protocol = 20 } },
    };
    const output_a_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .output = 10 } },
    };
    const output_b_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .output = 20 } },
    };

    const input_simplexes = [_]Simplex{.{ .vertices = &input_vertices }};
    const protocol_simplexes = [_]Simplex{
        .{ .vertices = &protocol_a_vertices },
        .{ .vertices = &protocol_b_vertices },
    };
    const output_simplexes = [_]Simplex{
        .{ .vertices = &output_a_vertices },
        .{ .vertices = &output_b_vertices },
    };
    const reachable_protocols = [_]usize{ 0, 1 };
    const carried_outputs = [_]usize{ 0, 1 };
    const execution_edges = [_]ExecutionEdge{
        .{ .input_index = 0, .protocol_indices = &reachable_protocols },
    };
    const task_edges = [_]CarrierEdge{
        .{ .input_index = 0, .output_indices = &carried_outputs },
    };
    const decisions = [_]ProtocolDecisionEdge{
        .{ .protocol_index = 0, .output_index = 0 },
        .{ .protocol_index = 1, .output_index = 1 },
    };
    const observations = [_]ExecutionObservation{
        .{ .input_index = 0, .protocol_index = 0 },
        .{ .input_index = 0, .protocol_index = 1 },
    };
    const execution = ExecutionCarrier{
        .input = .{ .simplexes = &input_simplexes },
        .protocol = .{ .simplexes = &protocol_simplexes },
        .carries = &execution_edges,
    };

    try validateExecutionDecisionMap(
        std.testing.allocator,
        execution,
        .{
            .input = .{ .simplexes = &input_simplexes },
            .output = .{ .simplexes = &output_simplexes },
            .carries = &task_edges,
        },
        &decisions,
    );
    try validateExecutionObservations(std.testing.allocator, execution, &observations);
}

test "execution observations must stay inside the bounded execution carrier" {
    const input_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .input = 1 } },
    };
    const protocol_a_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .protocol = 10 } },
    };
    const protocol_b_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .protocol = 20 } },
    };

    const input_simplexes = [_]Simplex{.{ .vertices = &input_vertices }};
    const protocol_simplexes = [_]Simplex{
        .{ .vertices = &protocol_a_vertices },
        .{ .vertices = &protocol_b_vertices },
    };
    const reachable_protocols = [_]usize{0};
    const execution_edges = [_]ExecutionEdge{
        .{ .input_index = 0, .protocol_indices = &reachable_protocols },
    };
    const observations = [_]ExecutionObservation{
        .{ .input_index = 0, .protocol_index = 1 },
    };

    try std.testing.expectError(error.ProtocolOutsideExecutionCarrier, validateExecutionObservations(
        std.testing.allocator,
        .{
            .input = .{ .simplexes = &input_simplexes },
            .protocol = .{ .simplexes = &protocol_simplexes },
            .carries = &execution_edges,
        },
        &observations,
    ));
}

test "execution decision map rejects task carrier escapes" {
    const input_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .input = 1 } },
    };
    const protocol_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .protocol = 10 } },
    };
    const output_ok_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .output = 10 } },
    };
    const output_bad_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .output = 20 } },
    };

    const input_simplexes = [_]Simplex{.{ .vertices = &input_vertices }};
    const protocol_simplexes = [_]Simplex{.{ .vertices = &protocol_vertices }};
    const output_simplexes = [_]Simplex{
        .{ .vertices = &output_ok_vertices },
        .{ .vertices = &output_bad_vertices },
    };
    const reachable_protocols = [_]usize{0};
    const carried_outputs = [_]usize{0};
    const execution_edges = [_]ExecutionEdge{
        .{ .input_index = 0, .protocol_indices = &reachable_protocols },
    };
    const task_edges = [_]CarrierEdge{
        .{ .input_index = 0, .output_indices = &carried_outputs },
    };
    const decisions = [_]ProtocolDecisionEdge{
        .{ .protocol_index = 0, .output_index = 1 },
    };

    try std.testing.expectError(error.DecisionOutsideCarrier, validateExecutionDecisionMap(
        std.testing.allocator,
        .{
            .input = .{ .simplexes = &input_simplexes },
            .protocol = .{ .simplexes = &protocol_simplexes },
            .carries = &execution_edges,
        },
        .{
            .input = .{ .simplexes = &input_simplexes },
            .output = .{ .simplexes = &output_simplexes },
            .carries = &task_edges,
        },
        &decisions,
    ));
}

test "one protocol decision must satisfy every input carrying that execution" {
    const input_a_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .input = 1 } },
    };
    const input_b_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .input = 2 } },
    };
    const protocol_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .protocol = 10 } },
    };
    const output_a_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .output = 1 } },
    };
    const output_b_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .output = 2 } },
    };

    const input_simplexes = [_]Simplex{
        .{ .vertices = &input_a_vertices },
        .{ .vertices = &input_b_vertices },
    };
    const protocol_simplexes = [_]Simplex{.{ .vertices = &protocol_vertices }};
    const output_simplexes = [_]Simplex{
        .{ .vertices = &output_a_vertices },
        .{ .vertices = &output_b_vertices },
    };
    const shared_protocol = [_]usize{0};
    const output_a = [_]usize{0};
    const output_b = [_]usize{1};
    const execution_edges = [_]ExecutionEdge{
        .{ .input_index = 0, .protocol_indices = &shared_protocol },
        .{ .input_index = 1, .protocol_indices = &shared_protocol },
    };
    const task_edges = [_]CarrierEdge{
        .{ .input_index = 0, .output_indices = &output_a },
        .{ .input_index = 1, .output_indices = &output_b },
    };
    const decisions = [_]ProtocolDecisionEdge{
        .{ .protocol_index = 0, .output_index = 0 },
    };

    try std.testing.expectError(error.DecisionOutsideCarrier, validateExecutionDecisionMap(
        std.testing.allocator,
        .{
            .input = .{ .simplexes = &input_simplexes },
            .protocol = .{ .simplexes = &protocol_simplexes },
            .carries = &execution_edges,
        },
        .{
            .input = .{ .simplexes = &input_simplexes },
            .output = .{ .simplexes = &output_simplexes },
            .carries = &task_edges,
        },
        &decisions,
    ));
}

test "decision map rejects outputs outside the carrier" {
    const input_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .input = 1 } },
    };
    const output_ok_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .output = 1 } },
    };
    const output_bad_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .output = 9 } },
    };
    const protocol_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .protocol = 1 } },
    };

    const input_simplexes = [_]Simplex{.{ .vertices = &input_vertices }};
    const output_simplexes = [_]Simplex{
        .{ .vertices = &output_ok_vertices },
        .{ .vertices = &output_bad_vertices },
    };
    const protocol_simplexes = [_]Simplex{.{ .vertices = &protocol_vertices }};
    const carried = [_]usize{0};
    const edges = [_]CarrierEdge{.{ .input_index = 0, .output_indices = &carried }};
    const decisions = [_]DecisionEdge{.{
        .input_index = 0,
        .protocol_index = 0,
        .output_index = 1,
    }};

    try std.testing.expectError(error.DecisionOutsideCarrier, validateDecisionMap(
        std.testing.allocator,
        .{ .simplexes = &protocol_simplexes },
        .{
            .input = .{ .simplexes = &input_simplexes },
            .output = .{ .simplexes = &output_simplexes },
            .carries = &edges,
        },
        &decisions,
    ));
}

test "decision map rejects incompatible outputs for a shared protocol view" {
    const input_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .input = 0 } },
    };
    const shared_protocol_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .protocol = 7 } },
    };
    const left_protocol_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .protocol = 7 } },
        .{ .color = .{ .process = 1 }, .label = .{ .protocol = 10 } },
    };
    const right_protocol_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .protocol = 7 } },
        .{ .color = .{ .process = 1 }, .label = .{ .protocol = 20 } },
    };
    const ok_output_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .output = 1 } },
    };
    const bad_output_vertices = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .output = 2 } },
    };

    const input_simplexes = [_]Simplex{.{ .vertices = &input_vertices }};
    const protocol_simplexes = [_]Simplex{
        .{ .vertices = &shared_protocol_vertices },
        .{ .vertices = &left_protocol_vertices },
        .{ .vertices = &right_protocol_vertices },
    };
    const output_simplexes = [_]Simplex{
        .{ .vertices = &ok_output_vertices },
        .{ .vertices = &bad_output_vertices },
    };
    const carried = [_]usize{ 0, 1 };
    const carrier_edges = [_]CarrierEdge{.{ .input_index = 0, .output_indices = &carried }};
    const decisions = [_]DecisionEdge{
        .{ .input_index = 0, .protocol_index = 0, .output_index = 0 },
        .{ .input_index = 0, .protocol_index = 1, .output_index = 0 },
        .{ .input_index = 0, .protocol_index = 2, .output_index = 1 },
    };

    try std.testing.expectError(error.IncompatibleDecisionOnSharedView, validateDecisionMap(
        std.testing.allocator,
        .{ .simplexes = &protocol_simplexes },
        .{
            .input = .{ .simplexes = &input_simplexes },
            .output = .{ .simplexes = &output_simplexes },
            .carries = &carrier_edges,
        },
        &decisions,
    ));
}

test "toy approximate agreement task is representable and decidable" {
    const input_left = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .input = 0 } },
    };
    const input_right = [_]Vertex{
        .{ .color = .{ .process = 1 }, .label = .{ .input = 2 } },
    };
    const input_both = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .input = 0 } },
        .{ .color = .{ .process = 1 }, .label = .{ .input = 2 } },
    };
    const output_left = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .output = 0 } },
    };
    const output_right = [_]Vertex{
        .{ .color = .{ .process = 1 }, .label = .{ .output = 2 } },
    };
    const output_mid = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .output = 1 } },
        .{ .color = .{ .process = 1 }, .label = .{ .output = 1 } },
    };
    const protocol_left = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .protocol = 10 } },
    };
    const protocol_right = [_]Vertex{
        .{ .color = .{ .process = 1 }, .label = .{ .protocol = 20 } },
    };
    const protocol_both = [_]Vertex{
        .{ .color = .{ .process = 0 }, .label = .{ .protocol = 11 } },
        .{ .color = .{ .process = 1 }, .label = .{ .protocol = 21 } },
    };

    const input_simplexes = [_]Simplex{
        .{ .vertices = &input_left },
        .{ .vertices = &input_right },
        .{ .vertices = &input_both },
    };
    const output_simplexes = [_]Simplex{
        .{ .vertices = &output_left },
        .{ .vertices = &output_right },
        .{ .vertices = &output_mid },
    };
    const protocol_simplexes = [_]Simplex{
        .{ .vertices = &protocol_left },
        .{ .vertices = &protocol_right },
        .{ .vertices = &protocol_both },
    };

    const left_outputs = [_]usize{0};
    const right_outputs = [_]usize{1};
    const both_outputs = [_]usize{ 0, 1, 2 };
    const carrier_edges = [_]CarrierEdge{
        .{ .input_index = 0, .output_indices = &left_outputs },
        .{ .input_index = 1, .output_indices = &right_outputs },
        .{ .input_index = 2, .output_indices = &both_outputs },
    };
    const decisions = [_]DecisionEdge{
        .{ .input_index = 0, .protocol_index = 0, .output_index = 0 },
        .{ .input_index = 1, .protocol_index = 1, .output_index = 1 },
        .{ .input_index = 2, .protocol_index = 2, .output_index = 2 },
    };

    try validateDecisionMap(
        std.testing.allocator,
        .{ .simplexes = &protocol_simplexes },
        .{
            .input = .{ .simplexes = &input_simplexes },
            .output = .{ .simplexes = &output_simplexes },
            .carries = &carrier_edges,
        },
        &decisions,
    );
}
