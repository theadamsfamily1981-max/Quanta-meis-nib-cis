#include <stdio.h>
#include <bpf/libbpf.h>
#include <sys/stat.h>
#include <unistd.h>

int main() {
  struct bpf_object *obj = bpf_object__open_file("dataflow.bpf.o", NULL);
  if (!obj) { perror("open bpf"); return 1; }
  if (bpf_object__load(obj)) { perror("load bpf"); return 1; }

  // mark cgroup /sys/fs/cgroup/lanemax-data as data-plane
  struct bpf_map *m = bpf_object__find_map_by_name(obj, "data_cgroups");
  if (!m) { fprintf(stderr, "map not found\n"); return 1; }
  int mapfd = bpf_map__fd(m);

  struct stat st = {0};
  if (!stat("/sys/fs/cgroup/lanemax-data", &st)) {
    __u64 key = st.st_ino; __u8 val = 1;
    bpf_map_update_elem(mapfd, &key, &val, BPF_ANY);
  }

  // attach struct_ops
  struct bpf_map *op = bpf_object__find_map_by_name(obj, "ops");
  if (!op) { fprintf(stderr, "ops not found\n"); return 1; }
  int err = bpf_link__attach_struct_ops(op);
  if (err) { fprintf(stderr, "attach failed: %d\n", err); return 1; }

  puts("sched_ext scx_dataflow loaded");
  pause(); // keep pinned
  return 0;
}
