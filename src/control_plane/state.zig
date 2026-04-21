const std = @import("std");

const Allocator = std.mem.Allocator;

pub const ControllerMode = enum {
    training,
    rl,
    inference,
};

pub const JobType = enum {
    training,
    rl,
    inference,
};

pub const JobStatus = enum {
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
    started_at: i64,
    finished_at: ?i64,
    last_error: ?[]u8,
    cancel_requested: std.atomic.Value(u8),
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
            .status = .starting,
            .config_path = if (config_path) |path| try allocator.dupe(u8, path) else null,
            .run_id = if (run_id) |id| try allocator.dupe(u8, id) else null,
            .model_id = if (model_id) |id| try allocator.dupe(u8, id) else null,
            .resume_requested = resume_requested,
            .required_workers = required_workers,
            .started_at = std.time.timestamp(),
            .finished_at = null,
            .last_error = null,
            .cancel_requested = std.atomic.Value(u8).init(0),
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
    }

    pub fn setStatus(self: *Self, status: JobStatus) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.status = status;
        if (isTerminal(status)) {
            self.finished_at = std.time.timestamp();
        }
    }

    pub fn setFailed(self: *Self, err: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.last_error) |old| self.allocator.free(old);
        self.last_error = try self.allocator.dupe(u8, err);
        self.status = .failed;
        self.finished_at = std.time.timestamp();
    }

    pub fn requestCancel(self: *Self) void {
        self.cancel_requested.store(1, .release);
        self.mutex.lock();
        defer self.mutex.unlock();
        if (!isTerminal(self.status)) {
            self.status = .cancelling;
        }
    }

    pub fn isCancellationRequested(self: *Self) bool {
        return self.cancel_requested.load(.acquire) == 1;
    }

    pub fn setCancelled(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.status = .cancelled;
        self.finished_at = std.time.timestamp();
    }

    pub fn setTrainingProgress(self: *Self, epoch: usize, step: usize, total_steps: usize, loss: f32, learning_rate: f32) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.training.current_epoch = epoch;
        self.training.current_step = step;
        self.training.total_steps = total_steps;
        self.training.loss = loss;
        self.training.learning_rate = learning_rate;
    }

    pub fn setRLProgress(self: *Self, iteration: usize, total_iterations: usize, rollouts_requested: usize, rollouts_completed: usize, reward_mean: f32) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.rl.current_iteration = iteration;
        self.rl.total_iterations = total_iterations;
        self.rl.rollouts_requested = rollouts_requested;
        self.rl.rollouts_completed = rollouts_completed;
        self.rl.reward_mean = reward_mean;
    }

    pub fn setInferenceMetrics(self: *Self, metrics: InferenceMetrics) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.inference = metrics;
    }

    pub fn renderControllerJson(self: *Self, allocator: Allocator, connected_workers: usize, ready: bool, auth_enabled: bool) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const Response = struct {
            mode: []const u8,
            job_type: []const u8,
            status: []const u8,
            ready: bool,
            auth_enabled: bool,
            config_path: ?[]const u8,
            run_id: ?[]const u8,
            model_id: ?[]const u8,
            @"resume": bool,
            started_at: i64,
            finished_at: ?i64,
            workers_connected: usize,
            workers_required: ?usize,
            cancel_requested: bool,
            last_error: ?[]const u8,
        };

        return std.json.stringifyAlloc(allocator, Response{
            .mode = modeString(self.mode),
            .job_type = jobTypeString(self.job_type),
            .status = statusString(self.status),
            .ready = ready,
            .auth_enabled = auth_enabled,
            .config_path = self.config_path,
            .run_id = self.run_id,
            .model_id = self.model_id,
            .@"resume" = self.resume_requested,
            .started_at = self.started_at,
            .finished_at = self.finished_at,
            .workers_connected = connected_workers,
            .workers_required = self.required_workers,
            .cancel_requested = self.cancel_requested.load(.acquire) == 1,
            .last_error = self.last_error,
        }, .{});
    }

    pub fn renderJobJson(self: *Self, allocator: Allocator, connected_workers: usize) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const Response = struct {
            job_type: []const u8,
            status: []const u8,
            config_path: ?[]const u8,
            run_id: ?[]const u8,
            model_id: ?[]const u8,
            @"resume": bool,
            started_at: i64,
            finished_at: ?i64,
            workers_connected: usize,
            workers_required: ?usize,
            cancel_requested: bool,
            last_error: ?[]const u8,
        };

        return std.json.stringifyAlloc(allocator, Response{
            .job_type = jobTypeString(self.job_type),
            .status = statusString(self.status),
            .config_path = self.config_path,
            .run_id = self.run_id,
            .model_id = self.model_id,
            .@"resume" = self.resume_requested,
            .started_at = self.started_at,
            .finished_at = self.finished_at,
            .workers_connected = connected_workers,
            .workers_required = self.required_workers,
            .cancel_requested = self.cancel_requested.load(.acquire) == 1,
            .last_error = self.last_error,
        }, .{});
    }

    pub fn renderMetricsJson(self: *Self, allocator: Allocator, connected_workers: usize) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const Response = struct {
            mode: []const u8,
            workers_connected: usize,
            training: TrainingMetrics,
            rl: RLMetrics,
            inference: InferenceMetrics,
        };

        return std.json.stringifyAlloc(allocator, Response{
            .mode = modeString(self.mode),
            .workers_connected = connected_workers,
            .training = self.training,
            .rl = self.rl,
            .inference = self.inference,
        }, .{});
    }

    pub fn statusString(status: JobStatus) []const u8 {
        return switch (status) {
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
        };
    }

    pub fn jobTypeString(job_type: JobType) []const u8 {
        return switch (job_type) {
            .training => "training",
            .rl => "rl",
            .inference => "inference",
        };
    }

    fn isTerminal(status: JobStatus) bool {
        return switch (status) {
            .completed, .failed, .cancelled => true,
            else => false,
        };
    }
};
