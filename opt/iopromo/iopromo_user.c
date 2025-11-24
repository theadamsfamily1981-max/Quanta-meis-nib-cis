#include <bpf/libbpf.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

static volatile int stop=0;
static void on_sig(int s){(void)s; stop=1;}

struct evt { unsigned long long pid, ino, bytes; };

static int handle(void *ctx, void *data, size_t len){
  struct evt *e = data;
  // simplistic heuristic: tell agent to "promote" this FD if possible
  // (your agent maps ino->path and flips to cuFile + nvCOMP path)
  fprintf(stderr, "[iopromo] pid=%llu ino=%llu bytes=%llu\n", e->pid, e->ino, e->bytes);
  // Option: write to a FIFO your agent watches
  return 0;
}

int main(){
  struct bpf_object *obj = bpf_object__open_file("iopromo.bpf.o", NULL);
  if (!obj || bpf_object__load(obj)) { perror("load bpf"); return 1; }

  struct ring_buffer *rb = NULL;
  int map = bpf_object__find_map_fd_by_name(obj, "ring");
  if (map < 0) { fprintf(stderr,"ring map not found\n"); return 1; }
  rb = ring_buffer__new(map, handle, NULL, NULL);
  if (!rb) { fprintf(stderr,"ring buffer new failed\n"); return 1; }

  signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
  while(!stop) ring_buffer__poll(rb, 100);
  ring_buffer__free(rb);
  return 0;
}
