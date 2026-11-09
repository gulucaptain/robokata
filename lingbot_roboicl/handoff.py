"""Transfer server T GPUs 4–7 from the identified RoboTwin run to RoboICL."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import zipfile

GPUS = (4, 5, 6, 7)
OLD_ROOT = Path("/data/haoyuzhao/exps/train/zero-wam-robotwin-b")
OLD_LAUNCH = OLD_ROOT / "launches/rank106-window16-e200"
OLD_SCRIPT = OLD_LAUNCH / "robotwin_b/train_distributed.py"
OLD_HASH = "bebef63221e7dbb2d40c765fb928d75ca8e97a57830f4aaa633af78b459d6b68"
OLD_RUN = OLD_ROOT / "runs/rank106-window16-e200"
WORK = Path("/data/haoyuzhao/envs/video/zero-wam-ig10k/cache/roboicl-full-v1")


def query(fields):
    output = subprocess.check_output(
        ["nvidia-smi", fields, "--format=csv,noheader,nounits"], text=True, timeout=10
    )
    return csv.reader(output.splitlines(), skipinitialspace=True)


def gpu_state():
    devices = {
        int(index): (uuid, int(free), int(total))
        for index, uuid, free, total in query("--query-gpu=index,uuid,memory.free,memory.total")
        if int(index) in GPUS
    }
    if set(devices) != set(GPUS):
        raise RuntimeError("Expected physical GPUs 4–7")
    processes = {uuid: set() for uuid, _, _ in devices.values()}
    for row in query("--query-compute-apps=gpu_uuid,pid"):
        if len(row) == 2 and row[0] in processes:
            processes[row[0]].add(int(row[1]))
    return {index: (free, total, processes[uuid]) for index, (uuid, free, total) in devices.items()}


def old_rank(pid):
    proc = Path(f"/proc/{pid}")
    if proc.stat().st_uid != os.getuid():
        raise RuntimeError(f"PID {pid} belongs to a different user")
    argv = (proc / "cmdline").read_bytes().split(b"\0")
    if (os.fsencode(OLD_SCRIPT) not in argv or b"--run" not in argv
            or argv[argv.index(b"--run") + 1] != os.fsencode(OLD_RUN) or b"--resume" not in argv):
        raise RuntimeError(f"PID {pid} is not the expected RoboTwin run")
    env = (proc / "environ").read_bytes().split(b"\0")
    return int(next(item.split(b"=", 1)[1] for item in env if item.startswith(b"LOCAL_RANK=")))


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path, help="Fresh absolute RoboICL run directory")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if not args.run.is_absolute() or args.run.exists() or args.timeout < 1:
        parser.error("Use a fresh absolute run path and a positive timeout")
    if socket.gethostname() != "h200-node2" or Path("/proc/self").stat().st_uid != OLD_RUN.stat().st_uid:
        raise RuntimeError("Expected server T and the old run owner")
    if hashlib.sha256(OLD_SCRIPT.read_bytes()).hexdigest() != OLD_HASH:
        raise RuntimeError("Old trainer differs from the reviewed signal-handler version")

    lock = (root / "logs/handoff-4-7.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    subprocess.run([root / "publish_roboicl.sh", "--check"], check=True)
    subprocess.run([root / "run_roboicl.sh", "inspect", "--work", WORK], check=True)
    if args.run.exists():
        raise RuntimeError("Target run directory appeared during preflight")

    status = OLD_RUN / "checkpoint-status.json"
    checkpoint = OLD_RUN / "checkpoints/latest.pt"
    exit_status = OLD_LAUNCH / "resume-pool.exit"
    before = json.loads(status.read_text())["step"]
    exit_mtime = exit_status.stat().st_mtime_ns if exit_status.exists() else 0
    state = gpu_state()
    pids = {}
    for gpu, (_, _, members) in state.items():
        if len(members) != 1:
            raise RuntimeError(f"GPU {gpu} does not have exactly one expected old process: {members}")
        pid = next(iter(members))
        if old_rank(pid) != gpu - GPUS[0]:
            raise RuntimeError(f"GPU {gpu} does not match the expected old local rank")
        pids[gpu] = pid
    print(f"Verified old run at checkpoint step {before}; GPU worker PIDs {pids}", flush=True)
    if not args.execute:
        return

    fd = os.pidfd_open(pids[4])
    try:
        if any(
            members != {pids[gpu]} or old_rank(pids[gpu]) != gpu - GPUS[0]
            for gpu, (_, _, members) in gpu_state().items()
        ):
            raise RuntimeError("Old job or GPU occupancy changed before handoff")
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
    print("Requested save-and-exit at the next optimizer boundary", flush=True)

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        state = gpu_state()
        for gpu, (_, _, members) in state.items():
            if members - {pids[gpu]}:
                raise RuntimeError(f"Another job entered GPU {gpu}: {members - {pids[gpu]}}")
        if all(not members and free >= .95 * total for free, total, members in state.values()):
            if exit_status.exists() and exit_status.stat().st_mtime_ns > exit_mtime:
                if exit_status.read_text().strip() != "0":
                    raise RuntimeError("Old launcher returned failure after the stop request")
                after = json.loads(status.read_text())["step"]
                if after <= before:
                    raise RuntimeError("Old checkpoint did not advance after the stop request")
                with zipfile.ZipFile(checkpoint) as archive:
                    if not any(name.endswith("/data.pkl") for name in archive.namelist()):
                        raise RuntimeError("Old checkpoint is missing its Torch payload")
                print(f"Old run saved step {after}; starting RoboICL for 20 epochs on GPUs 4–7", flush=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="4,5,6,7", NGPU="4", ACCUMULATION="16")
                job = root / "job_roboicl.sh"
                os.execve(job, [str(job), str(WORK), str(args.run), "20"], env)
        time.sleep(.5)
    raise TimeoutError("Timed out waiting for a clean old exit and reclaimed GPU memory")


if __name__ == "__main__":
    main()
