# Quanta-meis-nib-cis

Research for quanta meis nib cis code

commit directly to main

## Smart layer components

The optional "smart" layer introduces adaptive scheduling, I/O promotion, and
thermal routing helpers. The repository includes reference source in the same
layout used on production hosts.

### sched_ext two-lane scheduler

Files under `opt/scx/` implement a minimal `sched_ext` policy that biases tasks
in `/sys/fs/cgroup/lanemax-data` toward longer run quanta while permitting fast
preemption for control-plane work.

Build prerequisites (Ubuntu/Debian):

```
sudo apt-get install -y clang llvm libbpf-dev linux-headers-$(uname -r)
```

Build the BPF object and loader from the repository root:

```
cd opt/scx
clang -O2 -g -target bpf -c dataflow.bpf.c -o dataflow.bpf.o
gcc -O2 -g load_dataflow.c -o load_dataflow -lbpf
```

Run the scheduler and mark data-plane tasks:

```
sudo mkdir -p /sys/fs/cgroup/lanemax-data
sudo ./load_dataflow &
```

Launch workloads intended for long quanta inside the `lanemax-data` cgroup.

### eBPF I/O promoter

The `opt/iopromo/` directory contains an eBPF program that watches for large
sequential reads and emits ring-buffer events so the LaneMax agent can shift
workloads to the GDS path.

Build the probe and user-space reader:

```
cd opt/iopromo
clang -O2 -g -target bpf -c iopromo.bpf.c -o iopromo.bpf.o
gcc -O2 -g iopromo_user.c -o iopromo_user -lbpf
```

Run the notifier (requires root privileges to attach the kprobe):

```
sudo ./iopromo_user
```

### Thermal-aware routing helper

`usr/local/bin/lm-thermald` is a lightweight helper that writes indices of GPUs
reporting temperatures ≥85°C to `/run/lanemax.hot`. Launch it as a service so
job placement logic can avoid overheated devices:

```
sudo install -m755 usr/local/bin/lm-thermald /usr/local/bin/lm-thermald
sudo /usr/local/bin/lm-thermald
```

Integrate the output file with the scheduler or job launcher to divert work
away from the listed GPUs until they cool down.
