# BTP literature and evidence notebook

Started: 16 September 2026. Target: manuscript completion by 30 November 2026.

This is the first documented review pass, not a completed systematic review or proof of novelty. The collection contains 14 paper records, plus CRSDUN supplementary material. Reading depth is recorded per paper; downloaded does not mean reviewed. Review only the claims supported at the recorded reading depth.

## Start here

1. [Review and comparison matrix](REVIEW.md): source facts, relevance, reading depth, next questions, and paper cards.
2. [Claims ledger](CLAIMS_LEDGER.md): stable claim IDs, sources and locators. Use these IDs when drafting.
3. [Research gap and decision log](GAP_AND_DECISIONS.md): hypotheses, competing explanations, and required experiments.
4. [Experiment journal](EXPERIMENT_JOURNAL.md): what we actually ran and what the results do and do not establish.
5. [References](references.bib): bibliography generated from checked metadata. Entries marked provisional require a final publication-metadata check.
6. [Search log and reading queue](SEARCH_LOG.md): scope, search coverage, omissions and priorities.
7. [Research plan](../JOURNAL_RESEARCH_PLAN.md): schedule and milestone gates.
8. [Dataset completeness and preprocessing audit](../dataset_audit_20260916/REPORT.md): verified local coverage, conversion checks and unresolved author-protocol differences.

## Evidence rules

- **PUBLISHED** means the source states or reports the claim. It is not an independent reproduction.
- **OBSERVED** means our local experiment or file inspection supports it; always include run/configuration and scope.
- **HYPOTHESIS** is an idea requiring a discriminating test, never a result.
- **OPEN** flags missing data, unresolved discrepancies, or insufficient review.
- Store exact source version and access date where available. `download_manifest.json` records successful downloads, failures, and SHA-256 hashes. `papers.json` is the structured source of the review cards and bibliography.
- Record negative and failed experiments, not only improvements. Keep checkpoints/configurations and denominators with metrics.
- Never compare scores from different datasets, splits, spectral preprocessing, training budgets or metric conventions as a direct improvement.
- Use original paraphrases in the manuscript. Recheck every final claim against its cited paper; do not cite an abstract for details not present there.
- Update this notebook during subsequent research work. No autonomous monitoring or scheduled updates have been configured.

## What this pass changed

CRSDUN already includes a static segmentation-loss-weight sweep in its supplement. The original FVgNET paper reports segmentation masks for 80% of 317 images, whereas CRSDUN describes a 306-scene train/test protocol after exclusions. These are unresolved provenance differences, not evidence of an error by either author group. We must obtain or document the processed dataset before claiming exact reproduction.

No source files under the project's synced `sources/` directory were edited. The archive's `literature/sources/` directory is newly created research material, separate from that synced directory.

Latest preprocessing decision: [public252-v1 resolution](../preprocessing_resolution/RESOLUTION.md), including source evidence, frozen settings and remaining author-release limitation.
