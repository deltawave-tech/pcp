pub const composition = @import("composition.zig");
pub const coverage = @import("coverage.zig");
pub const decision_search = @import("decision_search.zig");
pub const invariants = @import("invariants.zig");
pub const obstruction = @import("obstruction.zig");
pub const sim = @import("sim.zig");
pub const protocol_complex = @import("protocol_complex.zig");
pub const reporting = @import("reporting.zig");
pub const shrink = @import("shrink.zig");
pub const solver = @import("solver.zig");
pub const subdivision = @import("subdivision.zig");
pub const tasks = @import("tasks.zig");
pub const topology = @import("topology.zig");
pub const trace = @import("trace.zig");

pub const extract = struct {
    pub const common = @import("extract/common.zig");
    pub const decoupled_site = @import("extract/decoupled_site.zig");
    pub const federation_placement = @import("extract/federation_placement.zig");
    pub const milestone4 = @import("extract/milestone4.zig");
    pub const message_registry = @import("extract/message_registry.zig");
    pub const rda_merge = @import("extract/rda_merge.zig");
    pub const scheduling = @import("extract/scheduling.zig");
    pub const workload_contract = @import("extract/workload_contract.zig");
};

pub const fixtures = struct {
    pub const workload_contract = @import("fixtures/workload_contract.zig");
};

pub const models = struct {
    pub const decoupled_site = @import("models/decoupled_site.zig");
    pub const federation_placement = @import("models/federation_placement.zig");
    pub const gateway_scheduling = @import("models/gateway_scheduling.zig");
    pub const workload_contract = @import("models/workload_contract.zig");
};

pub const oracles = struct {
    pub const decoupled_merge = @import("oracles/decoupled_merge.zig");
    pub const federation = @import("oracles/federation.zig");
    pub const message_registry = @import("oracles/message_registry.zig");
    pub const scheduling = @import("oracles/scheduling.zig");
    pub const workload_contract = @import("oracles/workload_contract.zig");
};

test {
    _ = composition;
    _ = coverage;
    _ = decision_search;
    _ = invariants;
    _ = obstruction;
    _ = sim;
    _ = protocol_complex;
    _ = reporting;
    _ = shrink;
    _ = solver;
    _ = subdivision;
    _ = tasks;
    _ = topology;
    _ = trace;

    _ = extract.common;
    _ = extract.decoupled_site;
    _ = extract.federation_placement;
    _ = extract.milestone4;
    _ = extract.message_registry;
    _ = extract.rda_merge;
    _ = extract.scheduling;
    _ = extract.workload_contract;

    _ = fixtures.workload_contract;

    _ = models.decoupled_site;
    _ = models.federation_placement;
    _ = models.gateway_scheduling;
    _ = models.workload_contract;

    _ = oracles.decoupled_merge;
    _ = oracles.federation;
    _ = oracles.message_registry;
    _ = oracles.scheduling;
    _ = oracles.workload_contract;
}
