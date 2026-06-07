pub const MessageFamily = enum {
    worker_control,
    artifact_transfer,
    regular_training,
    decoupled_training,
    streaming_diloco,
    inference,
    rl,
    custom_extension,
    federation,
    unknown,
};

pub const MessageDirection = enum {
    controller_to_worker,
    worker_to_controller,
    supervisor_to_controller,
    controller_to_supervisor,
    gateway_to_hub,
    hub_to_gateway,
    internal,
};

pub const WorkerHandlerFamily = enum {
    graph,
    transfer,
    regular_training,
    decoupled_training,
    streaming_diloco,
    rl,
    inference,
    custom_extension,
    lifecycle,
    unknown,
};

pub const MessageEntry = struct {
    msg_type: []const u8,
    family: MessageFamily,
    direction: MessageDirection,
    worker_handler: WorkerHandlerFamily = .unknown,
};

pub const MessageFamilyCapability = struct {
    family: MessageFamily,
    capability: []const u8,
};

pub const FederationOperation = struct {
    name: []const u8,
    method: []const u8,
    path_pattern: []const u8,
    direction: MessageDirection,
};

pub const WorkerHandlerRegistration = struct {
    family: WorkerHandlerFamily,
    messages: []const []const u8,
};
