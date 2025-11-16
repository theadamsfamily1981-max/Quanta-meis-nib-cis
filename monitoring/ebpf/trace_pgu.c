/*
 * eBPF Tracer for PGU (Proof Generation Unit)
 *
 * Traces PGU execution with nanosecond precision:
 * - Entry/exit timestamps
 * - Formula hash for deduplication
 * - Cache hit/miss tracking
 * - p95/p99 latency calculation
 *
 * Hard gate: p95 attribution coverage ≥95% of wall-time
 *
 * Usage:
 *   bpftrace trace_pgu.c
 *   python scripts/collect_traces.py --component pgu
 */

#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

// PGU event structure
struct pgu_event {
    u64 timestamp_ns;
    u32 pid;
    u32 tid;
    u64 formula_hash;
    u8 cache_hit;
    u64 duration_ns;
    char comm[16];
};

// BPF maps
BPF_HASH(pgu_start, u64, u64);           // pid → start_timestamp
BPF_HASH(pgu_formula, u64, u64);         // pid → formula_hash
BPF_PERF_OUTPUT(pgu_events);

// Trace PGU entry
int trace_pgu_enter(struct pt_regs *ctx, u64 formula_hash) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();

    pgu_start.update(&pid_tgid, &ts);
    pgu_formula.update(&pid_tgid, &formula_hash);

    return 0;
}

// Trace PGU exit
int trace_pgu_exit(struct pt_regs *ctx, u8 cache_hit) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 *start_ts = pgu_start.lookup(&pid_tgid);

    if (!start_ts) {
        return 0;  // No matching entry
    }

    u64 end_ts = bpf_ktime_get_ns();
    u64 duration = end_ts - *start_ts;

    // Construct event
    struct pgu_event event = {};
    event.timestamp_ns = end_ts;
    event.pid = pid_tgid >> 32;
    event.tid = pid_tgid & 0xFFFFFFFF;
    event.cache_hit = cache_hit;
    event.duration_ns = duration;

    // Get formula hash
    u64 *formula = pgu_formula.lookup(&pid_tgid);
    if (formula) {
        event.formula_hash = *formula;
    }

    // Get process name
    bpf_get_current_comm(&event.comm, sizeof(event.comm));

    // Submit event
    pgu_events.perf_submit(ctx, &event, sizeof(event));

    // Cleanup
    pgu_start.delete(&pid_tgid);
    pgu_formula.delete(&pid_tgid);

    return 0;
}

/*
 * BPFTrace script (alternative to C program):
 *
 * uprobe:/path/to/pgu:pgu_lookup_enter
 * {
 *     @start[tid] = nsecs;
 *     @formula[tid] = arg0;  // formula_hash
 * }
 *
 * uprobe:/path/to/pgu:pgu_lookup_exit
 * {
 *     $duration = nsecs - @start[tid];
 *     $cache_hit = arg0;
 *
 *     printf("PGU: duration=%llu ns, cache_hit=%d, formula=%llx\n",
 *            $duration, $cache_hit, @formula[tid]);
 *
 *     @pgu_latency = hist($duration);
 *     @pgu_p95 = quantize($duration, 95);
 *
 *     delete(@start[tid]);
 *     delete(@formula[tid]);
 * }
 */
