const std = @import("std");

pub const Sampler = struct {
    prng: std.Random.DefaultPrng,

    pub fn init(seed: u64) Sampler {
        return .{
            .prng = std.Random.DefaultPrng.init(seed),
        };
    }

    /// Sample a token index from logits using temperature scaling.
    /// Returns the index of the selected token.
    ///
    /// logits: Raw output from the model (un-normalized).
    /// temperature:
    ///   - 0.0: Greedy sampling (Argmax)
    ///   - 1.0: Standard sampling
    ///   - >1.0: Flatter distribution (more random)
    ///   - <1.0: Sharper distribution (more focused)
    pub fn sample(self: *Sampler, logits: []const f32, temperature: f32) i64 {
        // 1. Handle Greedy Decoding (Temp ~ 0)
        if (temperature < 1e-6) {
            return self.argmax(logits);
        }

        // 2. Find Max Logit (for numerical stability)
        var max_logit: f32 = -std.math.inf(f32);
        for (logits) |l| {
            if (l > max_logit) max_logit = l;
        }

        // 3. Compute Exponentials and Sum (The denominator of Softmax)
        // We assume we can't allocate a full probability array on the heap
        // for every token generation to keep latency low, so we compute in two passes.
        var sum_exp: f32 = 0.0;
        for (logits) |l| {
            sum_exp += @exp((l - max_logit) / temperature);
        }

        // 4. Random Selection (Inverse Transform Sampling)
        const random_val = self.prng.random().float(f32); // [0.0, 1.0)
        var cumulative_prob: f32 = 0.0;

        // We scale the random threshold by sum_exp so we don't have to divide
        // every single logit, effectively doing the comparison in "exp space".
        const threshold = random_val * sum_exp;

        for (logits, 0..) |l, i| {
            cumulative_prob += @exp((l - max_logit) / temperature);
            if (cumulative_prob >= threshold) {
                return @intCast(i);
            }
        }

        // Fallback (rare floating point rounding errors)
        return @intCast(logits.len - 1);
    }

    /// Helper: Find index of maximum value
    fn argmax(self: *Sampler, logits: []const f32) i64 {
        _ = self;
        var max_val: f32 = -std.math.inf(f32);
        var max_idx: usize = 0;

        for (logits, 0..) |val, i| {
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        return @intCast(max_idx);
    }
};

test "Sampler Basic" {
    var sampler = Sampler.init(12345);

    // Case 1: Greedy
    const logits_greedy = [_]f32{ 1.0, 5.0, 2.0 };
    const token_greedy = sampler.sample(&logits_greedy, 0.0);
    try std.testing.expectEqual(@as(i64, 1), token_greedy);

    // Case 2: High Probability (Temp=1.0)
    // 10.0 is much larger than 1.0, effectively 100% prob
    const logits_obvious = [_]f32{ 1.0, 10.0, 1.0 };
    const token_obvious = sampler.sample(&logits_obvious, 1.0);
    try std.testing.expectEqual(@as(i64, 1), token_obvious);
}
