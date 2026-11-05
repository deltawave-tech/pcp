/// A message broker to easily send and receive messages between different services of pcp.
const std = @import("std");
const Allocator = std.mem.Allocator;
const message = @import("message.zig");

const NodeId = message.NodeId;
const ServiceId = message.ServiceId;
const MessageEnvelope = message.MessageEnvelope;

const ClientList = std.ArrayList(ClientProxy);
const ServiceRegistry = std.StringHashMap(*ClientList);

pub const Broker = struct {
    this_node: NodeId,
    node_registry: NodeRegistry,
    allocator: Allocator,

    const NodeRegistry = std.AutoHashMap(NodeId, ServiceRegistry);

    const Self = @This();

    pub fn init(allocator: Allocator, node: NodeId) Self {
        std.log.info("Init a new Broker for node {}", .{node});
        const service_registry = NodeRegistry.init(allocator);
        return Self{
            .this_node = node,
            .node_registry = service_registry,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        std.log.info("Deinit Broker for node {}", .{self.this_node});
        var iter_nodes = self.node_registry.iterator();
        while (iter_nodes.next()) |node_entry| {
            std.log.debug("Unregister services for node {}", .{node_entry.key_ptr.*});
            var service_registry = node_entry.value_ptr.*;
            var iter_client_lists = service_registry.iterator();
            while (iter_client_lists.next()) |service_entry| {
                std.log.debug("Removing clients for service {s} {*}", .{ service_entry.key_ptr.*, service_entry.value_ptr });
                const client_list = service_entry.value_ptr.*;
                for (client_list.items) |*client| {
                    std.log.debug("Deinit client {}", .{client});
                    client.deinit(self.allocator);
                }
                client_list.deinit();
                self.allocator.destroy(client_list);
            }
            service_registry.deinit();
        }

        self.node_registry.deinit();
    }

    pub fn register(self: *Self, node: NodeId, service: ServiceId) error{OutOfMemory}!void {
        std.log.info("Register a new service {s} on node {}", .{ service, node });
        const node_registry_ptr = try self.node_registry.getOrPut(node);
        if (!node_registry_ptr.found_existing) {
            node_registry_ptr.value_ptr.* = ServiceRegistry.init(self.allocator);
        }
        var service_registry: *ServiceRegistry = &node_registry_ptr.value_ptr.*;
        if (service_registry.contains(service)) {
            std.log.info("Service was already registered", .{});
            return;
        }
        const client_list = try self.allocator.create(ClientList);
        client_list.* = ClientList.init(self.allocator);
        try service_registry.put(service, client_list);
    }

    pub fn unregister(self: *Self, node: NodeId, service: ServiceId) void {
        _ = self;
        _ = node;
        _ = service;
    }

    pub fn discover(
        self: Self,
        recipient_node: NodeId,
        recipient_service: ServiceId,
    ) error{ NoSuchNode, NoSuchService, OutOfMemory }!ClientProxy {
        std.log.info("Discovery for service {s} on node {}", .{ recipient_service, recipient_node });
        var service_registry = self.node_registry.get(recipient_node) orelse return error.NoSuchNode;
        var proxies = service_registry.get(recipient_service) orelse return error.NoSuchService;
        std.log.debug("List length: {}", .{proxies.items.len});
        std.log.debug("Inspecting proxy list at {*}: {any}", .{ &proxies, proxies.items });
        const recipient = Participant{ .node = recipient_node, .service = recipient_service };

        for (proxies.items) |proxy| {
            std.log.debug("Checking proxy {any}", .{proxy});
            if (proxy.hasRecipient(recipient)) {
                return proxy;
            }
        }
        const shared_mem = try ClientProxy.SharedMemory.init(self.allocator, recipient);
        const proxy = ClientProxy{ .shared_memory = shared_mem };
        try proxies.append(proxy);
        std.log.debug("Added proxy to list: {any}", .{proxies.items});
        std.log.debug("List length: {}", .{proxies.items.len});
        return proxy;
    }
};

pub const Participant = struct {
    node: NodeId,
    service: ServiceId,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        node: NodeId,
        service: ServiceId,
    ) error{OutOfMemory}!Self {
        const copied_service = try allocator.dupe(u8, service);
        std.debug.print("{*}\n", .{copied_service});
        return Self{
            .node = node,
            .service = copied_service,
        };
    }

    pub fn clone(self: Self, allocator: Allocator) error{OutOfMemory}!Self {
        return Self.init(
            allocator,
            self.node,
            self.service,
        );
    }

    pub fn deinit(self: *Self, allocator: Allocator) void {
        allocator.free(self.service);
    }

    pub fn isEqual(self: Self, other: Self) bool {
        return self.node == other.node and std.mem.eql(u8, self.service, other.service);
    }
};

pub const ClientProxy = union(enum) {
    pub const SharedMemory = struct {
        recipient: Participant,

        pub fn init(allocator: Allocator, recipient: Participant) error{OutOfMemory}!SharedMemory {
            const cloned_recipient = try recipient.clone(allocator);
            return SharedMemory{ .recipient = cloned_recipient };
        }

        pub fn deinit(self: *SharedMemory, allocator: Allocator) void {
            self.recipient.deinit(allocator);
        }

        pub fn send(self: SharedMemory, msg: MessageEnvelope) void {
            _ = self;
            _ = msg;
        }
        pub fn receive(self: SharedMemory) MessageEnvelope {
            _ = self;
            return undefined;
        }
    };

    shared_memory: SharedMemory,

    const Self = @This();

    pub fn initSharedMemory(allocator: Allocator, recipient: Participant) error{OutOfMemory}!Self {
        std.log.info("Initialize a ClientProxy with SharedMemory to {any}", .{recipient});
        const proxy_inner = try ClientProxy.SharedMemory.init(allocator, recipient);
        return ClientProxy{ .shared_memory = proxy_inner };
    }

    pub fn deinit(self: *Self, allocator: Allocator) void {
        std.log.info("Deinit ClientProxy for recipient {}", .{switch (self.*) {
            inline else => |*proxy| proxy.recipient,
        }});
        switch (self.*) {
            inline else => |*proxy| proxy.deinit(allocator),
        }
    }

    pub fn send(self: Self, msg: MessageEnvelope) void {
        switch (self) {
            inline else => |proxy| proxy.send(msg),
        }
    }

    pub fn receive(self: Self) MessageEnvelope {
        return switch (self) {
            inline else => |proxy| proxy.receive(),
        };
    }

    pub fn hasRecipient(self: Self, participant: Participant) bool {
        return switch (self) {
            inline else => |proxy| proxy.recipient.isEqual(participant),
        };
    }
};

test "init and deinit" {
    const allocator = std.testing.allocator;
    var broker = Broker.init(allocator, 0);
    broker.deinit();
}

test "discover on a not registered node" {
    const allocator = std.testing.allocator;
    const my_node_id = 0;
    var broker = Broker.init(allocator, 0);
    defer broker.deinit();

    try std.testing.expectError(error.NoSuchNode, broker.discover(my_node_id + 1, "SomeService"));
}

test "discover on a not registered service" {
    std.testing.log_level = std.log.Level.debug;
    const allocator = std.testing.allocator;
    const my_node_id = 0;
    var broker = Broker.init(allocator, my_node_id);
    try broker.register(my_node_id, "ServiceA");
    defer broker.deinit();

    try std.testing.expectError(error.NoSuchService, broker.discover(my_node_id, "ServiceB"));

    try std.testing.expectEqual(1, broker.node_registry.count());
    const service_registry = broker.node_registry.get(my_node_id).?;
    try std.testing.expectEqual(1, service_registry.count());
    const client_list = service_registry.get("ServiceA").?;
    try std.testing.expectEqual(0, client_list.items.len);
}

test "init and deinit a Participant" {
    const allocator = std.testing.allocator;
    var participant = try Participant.init(allocator, 0, "ServiceA");
    defer participant.deinit(allocator);
}

test "init and deinit a ClientProxy" {
    const allocator = std.testing.allocator;
    const recipient = Participant{ .node = 0, .service = "A" };
    var proxy = try ClientProxy.initSharedMemory(allocator, recipient);
    defer proxy.deinit(allocator);
    try std.testing.expect(proxy.hasRecipient(recipient));
}

//test "Register a service in a ServiceRegistry" {
//    const allocator = std.testing.allocator;
//    const recipient = Participant{ .node = 0, .service = "A" };
//    const service_registry = ServiceRegistry
//}

test "register and discover" {
    const allocator = std.testing.allocator;
    const my_node_id = 0;
    const service_a = "ServiceA";

    var broker = Broker.init(allocator, my_node_id);
    defer broker.deinit();
    try broker.register(my_node_id, service_a);
    const proxy1 = try broker.discover(my_node_id, service_a);

    try std.testing.expectEqual(1, broker.node_registry.count());
    const service_registry = broker.node_registry.get(my_node_id).?;
    try std.testing.expectEqual(1, service_registry.count());
    const client_list = service_registry.get(service_a).?;
    try std.testing.expectEqual(1, client_list.items.len);
    const proxy2 = try broker.discover(my_node_id, service_a);
    try std.testing.expectEqual(proxy1, proxy2);

    //const msg = proxy1.constructMessage("text", 1, std.json.Value{ .string = "foo" });
    // proxy.send(msg);
    // const result = proxy.receive();
    // try message.expectEqualMessages(msg, result);
}
