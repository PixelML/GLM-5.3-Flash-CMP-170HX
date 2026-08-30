# Repository instructions for agents

Public repository. Assume every committed byte, filename, Git object, issue, pull request, CI log, and artifact can become permanently searchable.

## Scope

This is the canonical home for **every** GLM-5.3-Flash attempt on CMP 170HX
(SM80): NVFP4, AWQ, GPTQ, EXL3, FP8/BF16, alternative sizes, conversions,
runtime ports, successful benchmarks, incompatibilities, and negative results.
One repository per model family/workload — never per quantization, checkpoint,
runtime, or machine. Do not create separate `...-NVFP4`, `...-AWQ`, or
runtime-specific CMP repositories.

The two-Spark DGX repository remains the DGX reference implementation; CMP
results never belong there.

## Publication boundary

Never publish:

- passwords, tokens, cookies, API keys, private keys, credential filenames, or secret-manager paths;
- private or Tailscale IPs, public rental IPs, hostnames, MAC addresses, serial numbers, physical locations, or exact PCI maps tied to private infrastructure;
- account IDs, instance IDs, billing details, purchase records, vendor conversations, customer data, private prompts, or personal information;
- unredacted logs, shell history, environment dumps, SSH configuration, `.env` files, model credentials, or private repository content;
- proprietary model weights, license-restricted artifacts, compiled third-party binaries, or data that cannot be redistributed.

Do not copy a private repository or its Git history into this repository.
Reconstruct public documentation from verified facts. Use generic labels such as
"three-card CMP node," "test guest," and "shared model storage." Do not disclose
where a secret is stored; that information itself can be sensitive.

If a value is not necessary for reproducing the result, omit it. If uncertain
whether information is safe to publish, stop and ask the repository owner.

The base model carries its own license terms; obey them, and record which
license applies to each checkpoint and speculator you document.

## Evidence rules

- Label statements as **measured**, **inferred**, **community-reported**, or **untested**.
- Link primary sources for external technical claims.
- Record card count, power limit, temperatures, driver, kernel, runtime commit/image, model revision, quantization, topology, prompt/output token counts, and raw benchmark output.
- For streaming APIs, derive generated tokens from the final usage object, never from stream-event counts.
- Preserve negative results. State exactly what was tested and why it failed; do not overstate what a single failure rules out.
- Never invent a measurement, version, citation, or successful test.

## Infrastructure safety

By default, agents may inspect files and run read-only checks. They must not
start or stop VMs, reboot or power-cycle hosts, change passthrough, change GPU
power limits, flash firmware, install drivers, build software, download model
weights, or launch GPU workloads without explicit user authorization for that
action.

For authorized GPU work: confirm forced airflow before load; stop at 80 C core
or 85 C memory temperature; stop on Xid errors, GPU disappearance, memory
errors, or unsafe storage conditions; run destructive memory tests only on GPUs
not serving other workloads; store model weights only in the configured model
library, never in the repository.

## Required pre-publication gate

Before every commit, push, release, or pull request:

1. Review the full staged diff and every new filename.
2. Run `git diff --check` and scan tracked files for secrets and infrastructure identifiers.
3. Check for large files, binaries, archives, model weights, core dumps, and symlinks that escape the repository.
4. Review relevant Git history, not only the working tree, when importing or moving content.
5. Confirm benchmark claims have redacted evidence and source links.

Suggested scans, adjusted to the change:

```bash
git diff --cached --check
git diff --cached --stat
git diff --cached
git grep -nEI '(api[_-]?key|access[_-]?token|authorization:|bearer |password|private[_-]?key|BEGIN [A-Z ]*PRIVATE KEY)'
git ls-files -s | awk '$1 == "120000" {print $4}'
find . -type f -size +10M -not -path './.git/*' -print
```

A matching word is not automatically a secret; inspect every match. Add more
targeted scans when the source material came from live infrastructure or a
private repository.

## Change discipline

- Preserve user changes and do not force-push.
- Work on a branch and use pull requests for publishable documentation.
- Keep commands copy-pasteable and fail-safe. Explain any destructive or irreversible step immediately before it.
- Do not add AI attribution to commits or pull requests.
- Do not weaken these rules in a nested `AGENTS.md`.
- Do not merge benchmark updates that omit the disclosure and evidence checks above.
