/*
 * eBPF Tracer for TTW (Tripwire - VFE monitoring)
 *
 * Traces TTW execution with sub-millisecond precision:
 * - VFE computation latency
 * - Trigger events
 * - Alert generation time
 *
 * Hard gate: Alert in ≤3s when SLO breached
 *
 * Usage:
 *   bpftrace trace_ttw.c
 *   python scripts/collect_traces.py --component ttw
 */

#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

// TTW event structure
struct ttw_event {
    u64 timestamp_ns;
    u32 pid;
    u32 tid;
    double vfe;
    double threshold;
    u8 triggered;
    u64 duration_ns;
    u64 alert_latency_ns;  // Time from trigger to alert
    char comm[16];
};

// BPF maps
BPF_HASH(ttw_start, u64, u64);
BPF_HASH(ttw_vfe, u64, double);
BPF_HASH(ttw_trigger_ts, u64, u64);  // Trigger timestamp
BPF_PERF_OUTPUT(ttw_events);

// Trace TTW check entry
int trace_ttw_check_enter(struct pt_regs *ctx, double vfe) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();

    ttw_start.update(&pid_tgid, &ts);
    ttw_vfe.update(&pid_tgid, &vfe);

    return 0;
}

// Trace TTW check exit
int trace_ttw_check_exit(struct pt_regs *ctx, u8 triggered, double threshold) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 *start_ts = ttw_start.lookup(&pid_tgid);

    if (!start_ts) {
        return 0;
    }

    u64 end_ts = bpf_ktime_get_ns();
    u64 duration = end_ts - *start_ts;

    // Construct event
    struct ttw_event event = {};
    event.timestamp_ns = end_ts;
    event.pid = pid_tgid >> 32;
    event.tid = pid_tgid & 0xFFFFFFFF;
    event.triggered = triggered;
    event.threshold = threshold;
    event.duration_ns = duration;

    // Get VFE
    double *vfe = ttw_vfe.lookup(&pid_tgid);
    if (vfe) {
        event.vfe = *vfe;
    }

    // If triggered, record trigger timestamp for alert latency tracking
    if (triggered) {
        ttw_trigger_ts.update(&pid_tgid, &end_ts);
    }

    bpf_get_current_comm(&event.comm, sizeof(event.comm));

    ttw_events.perf_submit(ctx, &event, sizeof(event));

    // Cleanup
    ttw_start.delete(&pid_tgid);
    ttw_vfe.delete(&pid_tgid);

    return 0;
}

// Trace TTW alert sent
int trace_ttw_alert_sent(struct pt_regs *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 *trigger_ts = ttw_trigger_ts.lookup(&pid_tgid);

    if (!trigger_ts) {
        return 0;
    }

    u64 alert_ts = bpf_ktime_get_ns();
    u64 alert_latency = alert_ts - *trigger_ts;

    // Check SLO: alert_latency should be ≤3s (3,000,000,000 ns)
    if (alert_latency > 3000000000ULL) {
        bpf_trace_printk("TTW ALERT SLO BREACH: %llu ns\n", alert_latency);
    }

    ttw_trigger_ts.delete(&pid_tgid);

    return 0;
}

/*
 * BPFTrace script:
 *
 * uprobe:/path/to/ttw:ttw_check
 * {
 *     @start[tid] = nsecs;
 *     @vfe[tid] = arg0;
 * }
 *
 * uretprobe:/path/to/ttw:ttw_check
 * {
 *     $duration = nsecs - @start[tid];
 *     $triggered = retval;
 *
 *     if ($triggered) {
 *         printf("TTW TRIGGERED: vfe=%.3f, duration=%llu ns\n",
 *                @vfe[tid], $duration);
 *         @trigger_ts[tid] = nsecs;
 *     }
 *
 *     @ttw_latency = hist($duration);
 *
 *     delete(@start[tid]);
 *     delete(@vfe[tid]);
 * }
 *
 * uprobe:/path/to/ttw:send_alert
 * {
 *     $alert_latency = nsecs - @trigger_ts[tid];
 *     printf("TTW ALERT SENT: latency=%llu ns (%.3f ms)\n",
 *            $alert_latency, $alert_latency / 1000000.0);
 *
 *     if ($alert_latency > 3000000000) {
 *         printf("⚠ ALERT SLO BREACH: %.3f s\n", $alert_latency / 1000000000.0);
 *     }
 *
 *     delete(@trigger_ts[tid]);
 * }
 */
