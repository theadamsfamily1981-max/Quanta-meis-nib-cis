#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <linux/fs.h>

struct evt { __u64 pid; __u64 ino; __u64 bytes; };
struct { __uint(type, BPF_MAP_TYPE_RINGBUF); __uint(max_entries, 1<<20); } ring SEC(".maps");

SEC("kprobe/new_sync_read")
int BPF_KPROBE(kp_read, struct file *file, loff_t *ppos, char *buf, size_t len) {
  if (!file) return 0;
  if (len < (1<<20)) return 0; // only large reads >= 1MB
  struct dentry *de = BPF_CORE_READ(file, f_path.dentry);
  struct inode *ino = de ? BPF_CORE_READ(de, d_inode) : NULL;
  if (!ino) return 0;

  struct evt *e = bpf_ringbuf_reserve(&ring, sizeof(*e), 0);
  if (!e) return 0;
  e->pid   = bpf_get_current_pid_tgid() >> 32;
  e->ino   = BPF_CORE_READ(ino, i_ino);
  e->bytes = len;
  bpf_ringbuf_submit(e, 0);
  return 0;
}
char LICENSE[] SEC("license") = "GPL";
