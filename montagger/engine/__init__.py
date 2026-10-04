"""The tagger engine: profile-driven preprocessing, label parsing, category
resolution, dispatch overlays, and onnxruntime session management.

The preprocessing, label, category and dispatch behavior is ported line for
line from monbooru's internal/tagger package, so the same model folder (with
its optional profile.json / dispatch.json sidecars) produces the same tags
in both programs. Differences are documented in README.md.
"""
