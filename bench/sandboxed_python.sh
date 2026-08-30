#!/usr/bin/env bash
# Run an untrusted Python file in a minimal user-namespace jail.
# Read-only /usr, /bin, /lib, /lib64; writable tmpfs /tmp only; no network;
# PID namespace; 25 s wall clock, 20 s CPU, 512 MiB address-space.
set -eu

FILE=$1

exec unshare --user --map-root-user --mount --net --pid --fork --mount-proc /bin/sh -c '
set -eu
mount --make-rprivate /
mount -t tmpfs -o size=4m,mode=755 tmpfs /mnt
mkdir -p /mnt/usr /mnt/bin /mnt/lib /mnt/lib64 /mnt/tmp
mount --bind /usr /mnt/usr
mount --bind /bin /mnt/bin
mount --bind /lib /mnt/lib
mount --bind /lib64 /mnt/lib64
for d in usr bin lib lib64; do mount -o remount,ro,bind "/mnt/$d"; done
mount -t tmpfs -o size=64m,mode=1777 tmpfs /mnt/tmp
 cp "$1" /mnt/tmp/harness.py
 # The harness imports the candidate as "solution"; place it beside the
 # harness so "from solution import ..." resolves inside the chroot.
 SOLUTION="${2:-}"
 if [ -n "$SOLUTION" ]; then
   cp "$SOLUTION" /mnt/tmp/solution.py
 else
   : > /mnt/tmp/solution.py
 fi
exec chroot /mnt /bin/sh -c "
  cd /tmp
  ulimit -t 20
  ulimit -v 524288
 exec python3 -I -B /tmp/harness.py
"
' runner "$FILE"
