pub const EmbeddedCustomInferenceService = DisabledCustomService;
pub const EmbeddedCustomTrainingService = DisabledCustomService;

pub const DisabledCustomService = struct {
    pub fn deinit(_: *@This()) void {}
};

pub const service_type_custom_inference = "";
pub const service_type_custom_training = "";
pub const route_custom_internal_training_jobs = "";
pub const route_custom_predict = "";
pub const route_custom_go_no_go = "";
pub const route_custom_training_jobs = "";

pub const capabilities = [_][]const u8{};

pub fn isCustomInferenceServiceType(_: []const u8) bool {
    return false;
}

pub fn isCustomTrainingServiceType(_: []const u8) bool {
    return false;
}

pub fn appendReadinessRequirements(_: anytype, _: anytype) !void {}

pub fn startEmbeddedServices(_: anytype, _: anytype) !void {}
