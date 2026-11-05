const types = @import("message_registry_types.zig");

pub const capability_custom_extension = "message_family.custom_extension";

pub const family_capabilities = [_]types.MessageFamilyCapability{};
pub const custom_extension_worker_messages = [_][]const u8{};
pub const worker_handler_registry = [_]types.WorkerHandlerRegistration{};
pub const protocol_messages = [_]types.MessageEntry{};
pub const all_message_types = [_][]const u8{};

pub fn isGatewayQueuedWorkerResultMessage(_: []const u8) bool {
    return false;
}
