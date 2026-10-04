v0.2.0

- Multi-model tagging: any number of default models per upload, results kept
  side by side (never merged), pushes separated per model (create + enrich
  per extra model, source `montagger/<model>`, no cross-model dedup).
- Monbooru push modes: `montagger` (own tags) or `monbooru_builtin`
  (`autotag=true`, the only way to get `is_auto=1` labels - documented API
  limitation otherwise).
- Monbooru-style threshold dialogs: per-model global + per-category
  thresholds + per-category top-k + disabled categories (nested config,
  old flat keys still load).
- Queue overhaul: server-side pagination + search, per-job cancel, batch
  delete/cancel, optional unbounded queue (`max_pending = 0`).
- Memory control: idle model unloading, per-model unload button,
  one-click release (sessions + cached jobs + gc + malloc_trim), RSS in
  /health.
- UI: confidence bars and rating badges on tags, grouped copy, per-model
  txt download, history grouped by file across models.
