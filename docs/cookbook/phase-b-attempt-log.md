# Phase B attempt log — four-card llama.cpp smoke (2026-08-30)

Runtime: llama.cpp fork at commit 00699716 (CUDA build, sm_80).
Checkpoint: UD-IQ4_XS 5-shard GGUF, pinned HF revision 2975ab4,
156,822,111,075 bytes total, per-shard SHA-256 verified before attempts.

## Result

No serving result. Four bounded startup attempts (up to 40 min health
window each) all ended identically before tensor offload:

- loader reached the metadata/unused-tensor stage (~6 min in; the last
  line is always the final blk.45.nextn warning), then the process
  vanished with no error line and no kernel-visible kill in the guest;
- immediately after each death the guest NVIDIA device nodes were gone
  (nvidia-smi failing) and had to be recreated before the next attempt;
- guest kernel logs show no Xid, no OOM kill, no segfault/audit record.

## Classification

Environmental teardown of the GPU device namespace by something outside
the guest, on a short recurring cycle. Not evidence against the model,
quant, runtime, or SM80 kernels: tensor offload was never reached, so
the compatibility hypotheses of this gate remain untested.

## Next action

Identify and stop the host/hypervisor-side reset of this guest's GPU
passthrough; then re-run the single bounded smoke via
bench/run-experiment.sh. All raw attempt logs are preserved out of tree
under the run directory and are excluded from git.
