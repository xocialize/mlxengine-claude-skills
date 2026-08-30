# Publishing a CoreAI asset

## `coreai-community` on Hugging Face

**MEASURED.** Naming convention: `<Model>-CoreAI`.

**The identical-403 trap:** two separate prerequisites, both failing the same way —

1. Org **membership approval**
2. Adding the org to your **fine-grained token's scope**

Missing either produces an indistinguishable 403. Check both before debugging anything else.

**Fresh-download-verify before announcing.** Clone the published repo into a clean directory and
run parity against it — not against your local export.

---

## Model card contents

MEASURED-GOOD template, from `Real-ESRGAN-CoreAI`:

- **Asset table** — one row per variant: filename, source checkpoint, distinguishing config,
  size
- **Parity numbers, per variant** — fp16-on-device vs fp32 reference, min and mean PSNR, and how
  many/what inputs
- **Why this runtime** — the measured argument (energy, memory shape), not a claim
- **Usage** — exact input contract: name, shape, dtype, layout, value range, and the output
  shape. Plus the first-load specialization warning.
- **Reproducibility** — the export script, in the repo. "No opaque binaries."
- **Provenance and license** — upstream architecture and weights, their license, and what this
  repo redistributes

**The input contract is not optional.** A static-shape asset with an undocumented layout or
value range is unusable by anyone but its author.

---

## Durability

Xocialize GitHub is approved as the durable home for CoreAI package source and the skill stack
itself. Xocialize HF is available where `coreai-community` is not the right venue.

**Ask before the first public push** — publishing is outward-facing and per-repo.
