# Changelog

## [v0.2.0] - 2026-08-20
### Added
- Pairing is now manual, like monloader: montagger only restores stored credentials at startup; the operator clicks connect in the settings page and approves the request in monbooru. Denied/expired/rejected attempts drop the credentials and never re-offer silently.
- Pairing is one-to-many: one montagger can pair with several monbooru instances. Credentials (url/token/peer) live in montagger.toml as a `[[pairing]]` array instead of a sidecar file, tasks are keyed on (source, image_id) so identical ids never collide, and relay calls are routed back to the instance they came from via its peer secret. The settings monbooru section is gone - each pairing carries its own address.
- Settings page rebuilt as a monloader-style shell with a section rail; every field is editable and saved independently. monbooru url, callback url, via, backend, model_dir, activation, log level and webui_token joined the hot-tunable set (applied immediately, no restart); addr/state/resume are editable with a "takes effect after restart" note.
- hx-confirm prompts now go through an in-page dialog instead of the native browser confirm, with destructive actions (remove pairing, clear results/tasks) getting a red OK button and focus on cancel. Danger buttons use a solid red fill.
- Settings page drops the single monbooru url field (pairing entries own their addresses); callback url moved to paths, via to tagging.

## [v0.1.0] - 2026-08-19
### Added
- Initial release: local ONNX tagging pipeline (WD14-family models), relay buttons on monbooru, pairing flow, and a WebUI with dashboard and hot-tunable settings.