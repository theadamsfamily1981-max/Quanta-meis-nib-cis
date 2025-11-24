// minimal two-queue sched_ext policy: favor /sys/fs/cgroup/lanemax-data tasks
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include <scx/common.bpf.h>
#include <scx/sched_ext.bpf.h>

char LICENSE[] SEC("license") = "Dual BSD/GPL";

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __type(key, u64);      // cgroup inode id
  __type(value, __u8);   // 1 = data-plane
  __uint(max_entries, 64);
} data_cgroups SEC(".maps");

static __always_inline bool is_data_cgrp(struct task_struct *p) {
  struct cgroup_subsys_state *css = BPF_CORE_READ(p, sched_task_group, css);
  if (!css) return false;
  struct cgroup *cg = BPF_CORE_READ(css, cgroup);
  if (!cg) return false;
  u64 id = BPF_CORE_READ(cg, kn, id);
  __u8 *v = bpf_map_lookup_elem(&data_cgroups, &id);
  return v && *v == 1;
}

SEC(".struct_ops.link")
struct scx_ops ops;

SEC(".kprobe/scx_select_cpu")
int kprobe_noop(void) { return 0; }

SEC(".struct_ops")
struct scx_ops ops = {
  .select_cpu = (void *)0, // let core pick, we bias via prio/quanta
  .enqueue    = scx_bpf_enqueue,
  .dispatch   = scx_bpf_dispatch,
  .runnable   = scx_bpf_runnable,
  .dequeue    = scx_bpf_dequeue,
  .stopping   = scx_bpf_stopping,
  .quiescent  = scx_bpf_quiescent,
  .init_task  = (void *)0,
  .name       = "scx_dataflow",
  .flags      = SCX_OPS_ENQ_LAST, // FIFO-ish
};

SEC(".btf_decl_tag")
int __weak __tag_data_quanta;

SEC(".tracepoint/sched/sched_switch")
int tp_switch(struct trace_event_raw_sched_switch *ctx) {
  // nothing heavy; we keep it simple for safety
  return 0;
}

SEC(".fentry/sched_ext_pick_next_task")
int BPF_PROG(tune_quanta, struct task_struct *p, u64 *slice_ns) {
  if (is_data_cgrp(p)) {
    *slice_ns = 8 * 1000 * 1000;  // 8ms for data-plane
  } else {
    *slice_ns = 1 * 1000 * 1000;  // 1ms for control & others
  }
  return 0;
}
