# Experiments

- `manifests/` — resolved-manifest templates, one per planned experiment.
- `current/` — output directories of active runs.

Every run writes `resolved_manifest.json` into its own output directory before
doing any work (`lmm/manifest.py`). That file, not the launch command, is what
downstream analysis reads.

Name an experiment after **what it tests**, never after the hoped-for result.
`ExperimentManifest.validate()` rejects names containing `successful`,
`optimal`, `stable`, `proven` and similar.
