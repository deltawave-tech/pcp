const std = @import("std");

const Allocator = std.mem.Allocator;

pub const FragmentStrategy = enum {
    strided,
    balanced_tensor,
};

pub const TensorRole = enum {
    unknown,
    embedding,
    non_transformer,
    transformer,
};

pub const TensorSpec = struct {
    index: usize,
    value_count: usize,
    byte_count: usize,
    role: TensorRole = .unknown,

    pub fn fromElementCount(index: usize, value_count: usize, bytes_per_value: usize) !TensorSpec {
        return .{
            .index = index,
            .value_count = value_count,
            .byte_count = try std.math.mul(usize, value_count, bytes_per_value),
        };
    }
};

pub const TensorRange = struct {
    tensor_index: usize,
    value_count: usize,
    byte_count: usize,
    offset_bytes: usize,
    role: TensorRole,
};

pub const Fragment = struct {
    id: usize,
    offset: usize,
    tensors: []TensorRange,
    value_count: usize,
    byte_count: usize,

    pub fn containsTensor(self: Fragment, tensor_index: usize) bool {
        return self.findTensor(tensor_index) != null;
    }

    pub fn findTensor(self: Fragment, tensor_index: usize) ?TensorRange {
        for (self.tensors) |range| {
            if (range.tensor_index == tensor_index) return range;
        }
        return null;
    }
};

pub const FragmentPlan = struct {
    allocator: Allocator,
    strategy: FragmentStrategy,
    sync_interval_h: usize,
    fragments: []Fragment,

    pub fn deinit(self: *FragmentPlan) void {
        for (self.fragments) |fragment| {
            self.allocator.free(fragment.tensors);
        }
        self.allocator.free(self.fragments);
        self.* = undefined;
    }

    pub fn fragmentForTensor(self: FragmentPlan, tensor_index: usize) ?usize {
        for (self.fragments) |fragment| {
            if (fragment.containsTensor(tensor_index)) return fragment.id;
        }
        return null;
    }

    pub fn ownsTensor(self: FragmentPlan, tensor_index: usize) bool {
        return self.fragmentForTensor(tensor_index) != null;
    }

    pub fn offsetForFragment(self: FragmentPlan, fragment_id: usize) ?usize {
        if (fragment_id >= self.fragments.len) return null;
        return self.fragments[fragment_id].offset;
    }

    pub fn fragmentByteCount(self: FragmentPlan, fragment_id: usize) ?usize {
        if (fragment_id >= self.fragments.len) return null;
        return self.fragments[fragment_id].byte_count;
    }

    pub fn fragmentValueCount(self: FragmentPlan, fragment_id: usize) ?usize {
        if (fragment_id >= self.fragments.len) return null;
        return self.fragments[fragment_id].value_count;
    }

    pub fn validateAgainst(self: FragmentPlan, tensors: []const TensorSpec) !void {
        for (tensors) |tensor| {
            var count: usize = 0;
            for (self.fragments) |fragment| {
                for (fragment.tensors) |range| {
                    if (range.tensor_index == tensor.index) count += 1;
                }
            }

            if (count == 0) return error.TensorNotAssigned;
            if (count > 1) return error.TensorAssignedMultipleTimes;
        }
    }
};

pub fn buildFragmentPlan(
    allocator: Allocator,
    strategy: FragmentStrategy,
    tensors: []const TensorSpec,
    num_fragments: usize,
    sync_interval_h: usize,
) !FragmentPlan {
    return switch (strategy) {
        .strided => buildStrided(allocator, tensors, num_fragments, sync_interval_h),
        .balanced_tensor => buildBalancedTensor(allocator, tensors, num_fragments, sync_interval_h),
    };
}

pub fn buildStrided(
    allocator: Allocator,
    tensors: []const TensorSpec,
    num_fragments: usize,
    sync_interval_h: usize,
) !FragmentPlan {
    try validateInputs(tensors, num_fragments, sync_interval_h);

    const builders = try initBuilders(allocator, num_fragments);
    defer deinitBuilders(builders);

    for (tensors) |tensor| {
        const fragment_id = tensor.index % num_fragments;
        try builders[fragment_id].append(tensor);
    }

    return finishPlan(allocator, .strided, tensors, builders, sync_interval_h);
}

pub fn buildBalancedTensor(
    allocator: Allocator,
    tensors: []const TensorSpec,
    num_fragments: usize,
    sync_interval_h: usize,
) !FragmentPlan {
    try validateInputs(tensors, num_fragments, sync_interval_h);

    const sorted = try allocator.dupe(TensorSpec, tensors);
    defer allocator.free(sorted);
    std.mem.sort(TensorSpec, sorted, {}, tensorSizeDesc);

    const builders = try initBuilders(allocator, num_fragments);
    defer deinitBuilders(builders);

    for (sorted) |tensor| {
        const fragment_id = smallestFragment(builders);
        try builders[fragment_id].append(tensor);
    }

    return finishPlan(allocator, .balanced_tensor, tensors, builders, sync_interval_h);
}

const FragmentBuilder = struct {
    tensors: std.ArrayList(TensorSpec),
    value_count: usize = 0,
    byte_count: usize = 0,

    fn append(self: *FragmentBuilder, tensor: TensorSpec) !void {
        try self.tensors.append(tensor);
        self.value_count = try std.math.add(usize, self.value_count, tensor.value_count);
        self.byte_count = try std.math.add(usize, self.byte_count, tensor.byte_count);
    }

    fn deinit(self: *FragmentBuilder) void {
        self.tensors.deinit();
    }
};

fn initBuilders(allocator: Allocator, num_fragments: usize) ![]FragmentBuilder {
    const builders = try allocator.alloc(FragmentBuilder, num_fragments);
    errdefer allocator.free(builders);

    for (builders) |*builder| {
        builder.* = .{ .tensors = std.ArrayList(TensorSpec).init(allocator) };
    }

    return builders;
}

fn deinitBuilders(builders: []FragmentBuilder) void {
    const allocator = if (builders.len > 0) builders[0].tensors.allocator else return;
    for (builders) |*builder| builder.deinit();
    allocator.free(builders);
}

fn finishPlan(
    allocator: Allocator,
    strategy: FragmentStrategy,
    original_tensors: []const TensorSpec,
    builders: []FragmentBuilder,
    sync_interval_h: usize,
) !FragmentPlan {
    const fragments = try allocator.alloc(Fragment, builders.len);
    var initialized_fragments: usize = 0;
    errdefer {
        for (fragments[0..initialized_fragments]) |fragment| {
            allocator.free(fragment.tensors);
        }
        allocator.free(fragments);
    }

    for (builders, 0..) |*builder, fragment_id| {
        std.mem.sort(TensorSpec, builder.tensors.items, {}, tensorIndexAsc);

        var ranges = try allocator.alloc(TensorRange, builder.tensors.items.len);
        errdefer allocator.free(ranges);

        var offset_bytes: usize = 0;
        for (builder.tensors.items, 0..) |tensor, i| {
            ranges[i] = .{
                .tensor_index = tensor.index,
                .value_count = tensor.value_count,
                .byte_count = tensor.byte_count,
                .offset_bytes = offset_bytes,
                .role = tensor.role,
            };
            offset_bytes = try std.math.add(usize, offset_bytes, tensor.byte_count);
        }

        fragments[fragment_id] = .{
            .id = fragment_id,
            .offset = fragment_id % sync_interval_h,
            .tensors = ranges,
            .value_count = builder.value_count,
            .byte_count = builder.byte_count,
        };
        initialized_fragments += 1;
    }

    const plan = FragmentPlan{
        .allocator = allocator,
        .strategy = strategy,
        .sync_interval_h = sync_interval_h,
        .fragments = fragments,
    };
    try plan.validateAgainst(original_tensors);
    return plan;
}

fn validateInputs(tensors: []const TensorSpec, num_fragments: usize, sync_interval_h: usize) !void {
    if (num_fragments == 0) return error.InvalidFragmentCount;
    if (sync_interval_h == 0) return error.InvalidSyncInterval;

    for (tensors, 0..) |tensor, i| {
        if (tensor.byte_count == 0 and tensor.value_count != 0) return error.InvalidTensorSize;

        for (tensors[0..i]) |previous| {
            if (previous.index == tensor.index) return error.DuplicateTensorIndex;
        }
    }
}

fn smallestFragment(builders: []const FragmentBuilder) usize {
    var best: usize = 0;
    var best_size = builders[0].byte_count;

    for (builders[1..], 1..) |builder, fragment_id| {
        if (builder.byte_count < best_size) {
            best = fragment_id;
            best_size = builder.byte_count;
        }
    }

    return best;
}

fn tensorSizeDesc(_: void, lhs: TensorSpec, rhs: TensorSpec) bool {
    if (lhs.byte_count == rhs.byte_count) return lhs.index < rhs.index;
    return lhs.byte_count > rhs.byte_count;
}

fn tensorIndexAsc(_: void, lhs: TensorSpec, rhs: TensorSpec) bool {
    return lhs.index < rhs.index;
}

test "strided fragmentation assigns tensor index modulo fragment count" {
    const allocator = std.testing.allocator;
    const tensors = [_]TensorSpec{
        try TensorSpec.fromElementCount(0, 2, 4),
        try TensorSpec.fromElementCount(1, 3, 4),
        try TensorSpec.fromElementCount(2, 4, 4),
        try TensorSpec.fromElementCount(3, 5, 4),
        try TensorSpec.fromElementCount(4, 6, 4),
    };

    var plan = try buildStrided(allocator, &tensors, 3, 3);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 3), plan.fragments.len);
    try std.testing.expectEqual(@as(?usize, 0), plan.fragmentForTensor(0));
    try std.testing.expectEqual(@as(?usize, 1), plan.fragmentForTensor(1));
    try std.testing.expectEqual(@as(?usize, 2), plan.fragmentForTensor(2));
    try std.testing.expectEqual(@as(?usize, 0), plan.fragmentForTensor(3));
    try std.testing.expectEqual(@as(?usize, 1), plan.fragmentForTensor(4));
    try std.testing.expectEqual(@as(usize, 28), plan.fragmentByteCount(0).?);
    try std.testing.expectEqual(@as(usize, 0), plan.fragments[0].findTensor(0).?.offset_bytes);
    try std.testing.expectEqual(@as(usize, 8), plan.fragments[0].findTensor(3).?.offset_bytes);
}

test "balanced tensor fragmentation is deterministic and assigns every tensor once" {
    const allocator = std.testing.allocator;
    const tensors = [_]TensorSpec{
        try TensorSpec.fromElementCount(0, 25, 4),
        try TensorSpec.fromElementCount(1, 22, 4),
        try TensorSpec.fromElementCount(2, 3, 4),
        try TensorSpec.fromElementCount(3, 3, 4),
    };

    var plan = try buildBalancedTensor(allocator, &tensors, 2, 2);
    defer plan.deinit();

    try std.testing.expectEqual(@as(?usize, 0), plan.fragmentForTensor(0));
    try std.testing.expectEqual(@as(?usize, 1), plan.fragmentForTensor(1));
    try std.testing.expectEqual(@as(?usize, 1), plan.fragmentForTensor(2));
    try std.testing.expectEqual(@as(?usize, 0), plan.fragmentForTensor(3));
    try std.testing.expectEqual(@as(usize, 112), plan.fragmentByteCount(0).?);
    try std.testing.expectEqual(@as(usize, 100), plan.fragmentByteCount(1).?);
    try plan.validateAgainst(&tensors);

    var second = try buildBalancedTensor(allocator, &tensors, 2, 2);
    defer second.deinit();

    for (plan.fragments, 0..) |fragment, i| {
        try std.testing.expectEqual(fragment.byte_count, second.fragments[i].byte_count);
        try std.testing.expectEqual(fragment.tensors.len, second.fragments[i].tensors.len);
        for (fragment.tensors, 0..) |range, j| {
            try std.testing.expectEqual(range.tensor_index, second.fragments[i].tensors[j].tensor_index);
        }
    }
}

test "fragment offsets cover every sync step when P equals H" {
    const allocator = std.testing.allocator;
    const tensors = [_]TensorSpec{
        try TensorSpec.fromElementCount(10, 1, 4),
        try TensorSpec.fromElementCount(11, 1, 4),
        try TensorSpec.fromElementCount(12, 1, 4),
    };

    var plan = try buildBalancedTensor(allocator, &tensors, 3, 3);
    defer plan.deinit();

    try std.testing.expectEqual(@as(?usize, 0), plan.offsetForFragment(0));
    try std.testing.expectEqual(@as(?usize, 1), plan.offsetForFragment(1));
    try std.testing.expectEqual(@as(?usize, 2), plan.offsetForFragment(2));
}

test "fragment planning rejects duplicate tensor indices" {
    const allocator = std.testing.allocator;
    const tensors = [_]TensorSpec{
        try TensorSpec.fromElementCount(7, 1, 4),
        try TensorSpec.fromElementCount(7, 2, 4),
    };

    try std.testing.expectError(error.DuplicateTensorIndex, buildBalancedTensor(allocator, &tensors, 2, 2));
}
