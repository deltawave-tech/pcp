const std = @import("std");

const Allocator = std.mem.Allocator;

pub const ControllerMode = enum {
    training,
    rl,
    inference,
    custom_inference,
    custom_training,
};

pub const JobType = enum {
    training,
    rl,
    inference,
    custom_inference,
    custom_training,
};

pub const JobStatus = enum {
    idle,
    queued,
    starting,
    initializing,
    waiting_for_workers,
    running,
    completed,
    failed,
    cancelling,
    cancelled,
};

pub const TrainingMetrics = struct {
    current_epoch: usize = 0,
    current_step: usize = 0,
    total_steps: usize = 0,
    loss: f32 = 0.0,
    learning_rate: f32 = 0.0,
};

pub const RLMetrics = struct {
    current_iteration: usize = 0,
    total_iterations: usize = 0,
    rollouts_requested: usize = 0,
    rollouts_completed: usize = 0,
    reward_mean: f32 = 0.0,
};

pub const InferenceMetrics = struct {
    total_requests: u64 = 0,
    active_requests: u64 = 0,
    queued_requests: u64 = 0,
    completed_requests: u64 = 0,
    failed_requests: u64 = 0,
    tokens_generated: u64 = 0,
    prompt_tokens: u64 = 0,
    ttft_total_ms: u64 = 0,
    ttft_count: u64 = 0,
    cache_hits: u64 = 0,
    cache_misses: u64 = 0,
};

pub const RuntimeSnapshot = struct {
    workers_connected: usize,
    workers_ready: usize = 0,
    workers_available: usize = 0,
    health_status: []const u8 = "starting",
    ready: bool = false,
    readiness_phase: ?[]const u8 = null,
    readiness_detail: ?[]const u8 = null,
    worker_class: ?[]const u8 = null,
    target_arch: ?[]const u8 = null,
    reserved_workers: ?usize = null,
    max_workers: ?usize = null,
};

pub const ChangeHook = struct {
    ctx: *anyopaque,
    on_change: *const fn (ctx: *anyopaque) void,
};

pub const ControllerState = struct {
    allocator: Allocator,
    mutex: std.Thread.Mutex,
    mode: ControllerMode,
    job_type: JobType,
    status: JobStatus,
    config_path: ?[]u8,
    run_id: ?[]u8,
    model_id: ?[]u8,
    resume_requested: bool,
    required_workers: ?usize,
    submitted_at: ?i64,
    started_at: i64,
    finished_at: ?i64,
    last_error: ?[]u8,
    readiness_phase: ?[]u8,
    readiness_detail: ?[]u8,
    cancel_requested: std.atomic.Value(u8),
    change_hook: ?ChangeHook,
    training: TrainingMetrics,
    rl: RLMetrics,
    inference: InferenceMetrics,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        mode: ControllerMode,
        job_type: JobType,
        config_path: ?[]const u8,
        run_id: ?[]const u8,
        model_id: ?[]const u8,
        resume_requested: bool,
        required_workers: ?usize,
    ) !Self {
        return Self{
            .allocator = allocator,
            .mutex = .{},
            .mode = mode,
            .job_type = job_type,
            .status = .idle,
            .config_path = if (config_path) |path| try allocator.dupe(u8, path) else null,
            .run_id = if (run_id) |id| try allocator.dupe(u8, id) else null,
            .model_id = if (model_id) |id| try allocator.dupe(u8, id) else null,
            .resume_requested = resume_requested,
            .required_workers = required_workers,
            .submitted_at = if (run_id != null) std.time.timestamp() else null,
            .started_at = std.time.timestamp(),
            .finished_at = null,
            .last_error = null,
            .readiness_phase = null,
            .readiness_detail = null,
            .cancel_requested = std.atomic.Value(u8).init(0),
            .change_hook = null,
            .training = .{},
            .rl = .{},
            .inference = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.config_path) |path| self.allocator.free(path);
        if (self.run_id) |id| self.allocator.free(id);
        if (self.model_id) |id| self.allocator.free(id);
        if (self.last_error) |err| self.allocator.free(err);
        if (self.readiness_phase) |phase| self.allocator.free(phase);
        if (self.readiness_detail) |detail| self.allocator.free(detail);
    }

    pub fn setStatus(self: *Self, status: JobStatus) void {
        var hook: ?ChangeHook = null;
        self.mutex.lock();
        self.status = status;
        if (isTerminal(status)) {
            self.finished_at = std.time.timestamp();
        } else if (status == .running and self.started_at == 0) {
            self.started_at = std.time.timestamp();
        }
        hook = self.change_hook;
        self.mutex.unlock();
        notifyChange(hook);
    }

    pub fn setFailed(self: *Self, err: []const u8) !void {
        var hook: ?ChangeHook = null;
        self.mutex.lock();
        if (self.last_error) |old| self.allocator.free(old);
        self.last_error = try self.allocator.dupe(u8, err);
        self.status = .failed;
        self.finished_at = std.time.timestamp();
        hook = self.change_hook;
        self.mutex.unlock();
        notifyChange(hook);
    }

    pub fn requestCancel(self: *Self) void {
        self.cancel_requested.store(1, .release);
        var hook: ?ChangeHook = null;
        self.mutex.lock();
        if (!isTerminal(self.status)) {
            self.status = .cancelling;
        }
        hook = self.change_hook;
        self.mutex.unlock();
        notifyChange(hook);
    }

    pub fn isCancellationRequested(self: *Self) bool {
        return self.cancel_requested.load(.acquire) == 1;
    }

    pub fn setCancelled(self: *Self) void {
        var hook: ?ChangeHook = null;
        self.mutex.lock();
        self.status = .cancelled;
        self.finished_at = std.time.timestamp();
        hook = self.change_hook;
        self.mutex.unlock();
        notifyChange(hook);
    }

    pub fn setReadinessPhase(self: *Self, phase: ?[]const u8, detail: ?[]const u8) !void {
        var hook: ?ChangeHook = null;
        self.mutex.lock();
        if (self.readiness_phase) |existing| self.allocator.free(existing);
        if (self.readiness_detail) |existing| self.allocator.free(existing);
        self.readiness_phase = if (phase) |value| try self.allocator.dupe(u8, value) else null;
        self.readiness_detail = if (detail) |value| try self.allocator.dupe(u8, value) else null;
        hook = self.change_hook;
        self.mutex.unlock();
        notifyChange(hook);
    }

    pub fn setChangeHook(self: *Self, hook: ?ChangeHook) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.change_hook = hook;
    }

    pub fn setTrainingProgress(self: *Self, epoch: usize, step: usize, total_steps: usize, loss: f32, learning_rate: f32) void {
        var hook: ?ChangeHook = null;
        self.mutex.lock();
        self.training.current_epoch = epoch;
        self.training.current_step = step;
        self.training.total_steps = total_steps;
        self.training.loss = loss;
        self.training.learning_rate = learning_rate;
        hook = self.change_hook;
        self.mutex.unlock();
        notifyChange(hook);
    }

    pub fn setRLProgress(self: *Self, iteration: usize, total_iterations: usize, rollouts_requested: usize, rollouts_completed: usize, reward_mean: f32) void {
        var hook: ?ChangeHook = null;
        self.mutex.lock();
        self.rl.current_iteration = iteration;
        self.rl.total_iterations = total_iterations;
        self.rl.rollouts_requested = rollouts_requested;
        self.rl.rollouts_completed = rollouts_completed;
        self.rl.reward_mean = reward_mean;
        hook = self.change_hook;
        self.mutex.unlock();
        notifyChange(hook);
    }

    pub fn setInferenceMetrics(self: *Self, metrics: InferenceMetrics) void {
        var hook: ?ChangeHook = null;
        self.mutex.lock();
        self.inference = metrics;
        hook = self.change_hook;
        self.mutex.unlock();
        notifyChange(hook);
    }

    pub fn resetForJob(
        self: *Self,
        run_id: ?[]const u8,
        model_id: ?[]const u8,
        resume_requested: bool,
        required_workers: ?usize,
        submitted_at: ?i64,
        initial_status: JobStatus,
    ) !void {
        var hook: ?ChangeHook = null;
        self.mutex.lock();
        defer {
            self.mutex.unlock();
            notifyChange(hook);
        }

        if (self.run_id) |value| self.allocator.free(value);
        self.run_id = if (run_id) |value| try self.allocator.dupe(u8, value) else null;

        if (self.model_id) |value| self.allocator.free(value);
        self.model_id = if (model_id) |value| try self.allocator.dupe(u8, value) else null;

        if (self.last_error) |value| self.allocator.free(value);
        self.last_error = null;

        if (self.readiness_phase) |value| self.allocator.free(value);
        self.readiness_phase = null;

        if (self.readiness_detail) |value| self.allocator.free(value);
        self.readiness_detail = null;

        self.resume_requested = resume_requested;
        self.required_workers = required_workers;
        self.submitted_at = submitted_at;
        self.started_at = if (initial_status == .queued) 0 else std.time.timestamp();
        self.finished_at = null;
        self.status = initial_status;
        self.cancel_requested.store(0, .release);
        self.training = .{};
        self.rl = .{};
        self.inference = .{};
        hook = self.change_hook;
    }

    pub fn resetToIdle(
        self: *Self,
        model_id: ?[]const u8,
        required_workers: ?usize,
    ) !void {
        try self.resetForJob(null, model_id, false, required_workers, null, .idle);
    }

    pub fn renderControllerJson(self: *Self, allocator: Allocator, connected_workers: usize, ready: bool, auth_enabled: bool) ![]u8 {
        return self.renderControllerJsonWithRuntime(allocator, .{
            .workers_connected = connected_workers,
            .workers_ready = connected_workers,
            .workers_available = connected_workers,
            .health_status = if (ready) "ok" else "starting",
            .ready = ready,
        }, auth_enabled);
    }

    pub fn renderControllerJsonWithRuntime(self: *Self, allocator: Allocator, runtime: RuntimeSnapshot, auth_enabled: bool) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const Response = struct {
            mode: []const u8,
            job_type: []const u8,
            status: []const u8,
            ready: bool,
            health_status: []const u8,
            auth_enabled: bool,
            config_path: ?[]const u8,
            run_id: ?[]const u8,
            model_id: ?[]const u8,
            @"resume": bool,
            submitted_at: ?i64,
            started_at: i64,
            finished_at: ?i64,
            workers_connected: usize,
            workers_ready: usize,
            workers_available: usize,
            workers_dispatchable: usize,
            workers_required: ?usize,
            readiness_phase: ?[]const u8,
            readiness_detail: ?[]const u8,
            worker_class: ?[]const u8,
            target_arch: ?[]const u8,
            reserved_workers: ?usize,
            max_workers: ?usize,
            cancel_requested: bool,
            last_error: ?[]const u8,
        };

        return std.json.stringifyAlloc(allocator, Response{
            .mode = modeString(self.mode),
            .job_type = jobTypeString(self.job_type),
            .status = statusString(self.status),
            .ready = runtime.ready,
            .health_status = runtime.health_status,
            .auth_enabled = auth_enabled,
            .config_path = self.config_path,
            .run_id = self.run_id,
            .model_id = self.model_id,
            .@"resume" = self.resume_requested,
            .submitted_at = self.submitted_at,
            .started_at = self.started_at,
            .finished_at = self.finished_at,
            .workers_connected = runtime.workers_connected,
            .workers_ready = runtime.workers_ready,
            .workers_available = runtime.workers_available,
            .workers_dispatchable = runtime.workers_available,
            .workers_required = self.required_workers,
            .readiness_phase = runtime.readiness_phase orelse self.readiness_phase,
            .readiness_detail = runtime.readiness_detail orelse self.readiness_detail,
            .worker_class = runtime.worker_class,
            .target_arch = runtime.target_arch,
            .reserved_workers = runtime.reserved_workers,
            .max_workers = runtime.max_workers,
            .cancel_requested = self.cancel_requested.load(.acquire) == 1,
            .last_error = self.last_error,
        }, .{});
    }

    pub fn renderJobJson(self: *Self, allocator: Allocator, connected_workers: usize) ![]u8 {
        return self.renderJobJsonWithRuntime(allocator, .{
            .workers_connected = connected_workers,
            .workers_ready = connected_workers,
            .workers_available = connected_workers,
            .health_status = if (connected_workers > 0) "ok" else "starting",
            .ready = connected_workers >= (self.required_workers orelse 1),
        });
    }

    pub fn renderJobJsonWithRuntime(self: *Self, allocator: Allocator, runtime: RuntimeSnapshot) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const Response = struct {
            job_type: []const u8,
            status: []const u8,
            health_status: []const u8,
            config_path: ?[]const u8,
            run_id: ?[]const u8,
            model_id: ?[]const u8,
            @"resume": bool,
            submitted_at: ?i64,
            started_at: i64,
            finished_at: ?i64,
            workers_connected: usize,
            workers_ready: usize,
            workers_available: usize,
            workers_dispatchable: usize,
            workers_required: ?usize,
            ready: bool,
            readiness_phase: ?[]const u8,
            readiness_detail: ?[]const u8,
            worker_class: ?[]const u8,
            target_arch: ?[]const u8,
            reserved_workers: ?usize,
            max_workers: ?usize,
            cancel_requested: bool,
            last_error: ?[]const u8,
        };

        return std.json.stringifyAlloc(allocator, Response{
            .job_type = jobTypeString(self.job_type),
            .status = statusString(self.status),
            .health_status = runtime.health_status,
            .config_path = self.config_path,
            .run_id = self.run_id,
            .model_id = self.model_id,
            .@"resume" = self.resume_requested,
            .submitted_at = self.submitted_at,
            .started_at = self.started_at,
            .finished_at = self.finished_at,
            .workers_connected = runtime.workers_connected,
            .workers_ready = runtime.workers_ready,
            .workers_available = runtime.workers_available,
            .workers_dispatchable = runtime.workers_available,
            .workers_required = self.required_workers,
            .ready = runtime.ready,
            .readiness_phase = runtime.readiness_phase orelse self.readiness_phase,
            .readiness_detail = runtime.readiness_detail orelse self.readiness_detail,
            .worker_class = runtime.worker_class,
            .target_arch = runtime.target_arch,
            .reserved_workers = runtime.reserved_workers,
            .max_workers = runtime.max_workers,
            .cancel_requested = self.cancel_requested.load(.acquire) == 1,
            .last_error = self.last_error,
        }, .{});
    }

    pub fn renderMetricsJson(self: *Self, allocator: Allocator, connected_workers: usize) ![]u8 {
        return self.renderMetricsJsonWithRuntime(allocator, .{
            .workers_connected = connected_workers,
            .workers_ready = connected_workers,
            .workers_available = connected_workers,
            .health_status = if (connected_workers > 0) "ok" else "starting",
            .ready = connected_workers >= (self.required_workers orelse 1),
        });
    }

    pub fn renderMetricsJsonWithRuntime(self: *Self, allocator: Allocator, runtime: RuntimeSnapshot) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const Response = struct {
            mode: []const u8,
            workers_connected: usize,
            workers_ready: usize,
            workers_available: usize,
            workers_dispatchable: usize,
            training: TrainingMetrics,
            rl: RLMetrics,
            inference: InferenceMetrics,
        };

        return std.json.stringifyAlloc(allocator, Response{
            .mode = modeString(self.mode),
            .workers_connected = runtime.workers_connected,
            .workers_ready = runtime.workers_ready,
            .workers_available = runtime.workers_available,
            .workers_dispatchable = runtime.workers_available,
            .training = self.training,
            .rl = self.rl,
            .inference = self.inference,
        }, .{});
    }

    pub fn renderReadyJson(self: *Self, allocator: Allocator, runtime: RuntimeSnapshot) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        return std.json.stringifyAlloc(allocator, .{
            .ready = runtime.ready,
            .status = statusString(self.status),
            .health_status = runtime.health_status,
            .workers_connected = runtime.workers_connected,
            .workers_ready = runtime.workers_ready,
            .workers_available = runtime.workers_available,
            .workers_dispatchable = runtime.workers_available,
            .workers_required = self.required_workers,
            .readiness_phase = runtime.readiness_phase orelse self.readiness_phase,
            .readiness_detail = runtime.readiness_detail orelse self.readiness_detail,
            .worker_class = runtime.worker_class,
            .target_arch = runtime.target_arch,
            .reserved_workers = runtime.reserved_workers,
            .max_workers = runtime.max_workers,
            .last_error = self.last_error,
        }, .{});
    }

    pub fn statusString(status: JobStatus) []const u8 {
        return switch (status) {
            .idle => "idle",
            .queued => "queued",
            .starting => "starting",
            .initializing => "initializing",
            .waiting_for_workers => "waiting_for_workers",
            .running => "running",
            .completed => "completed",
            .failed => "failed",
            .cancelling => "cancelling",
            .cancelled => "cancelled",
        };
    }

    pub fn modeString(mode: ControllerMode) []const u8 {
        return switch (mode) {
            .training => "training",
            .rl => "rl",
            .inference => "inference",
            .custom_inference => "custom_inference",
            .custom_training => "custom_training",
        };
    }

    pub fn jobTypeString(job_type: JobType) []const u8 {
        return switch (job_type) {
            .training => "training",
            .rl => "rl",
            .inference => "inference",
            .custom_inference => "custom_inference",
            .custom_training => "custom_training",
        };
    }

    fn isTerminal(status: JobStatus) bool {
        return switch (status) {
            .completed, .failed, .cancelled => true,
            else => false,
        };
    }

    fn notifyChange(hook: ?ChangeHook) void {
        if (hook) |cb| {
            cb.on_change(cb.ctx);
        }
    }
};
