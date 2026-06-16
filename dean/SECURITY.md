# Security Checklist

This project is designed to run offline-first for sensitive university spreadsheets.

## Application Defaults

- Cloud APIs are not used.
- Telemetry is not implemented.
- Analytics are not implemented.
- Remote logging is not implemented.
- Local LLM fallback is disabled by default.
- Local authentication is used; no external identity provider is configured.
- Passwords are stored only as salted PBKDF2-SHA256 hashes.
- Viewer, Editor, and Admin roles limit export and administrative access.
- Ollama access is hard-coded to `http://localhost:11434`.
- Remote Ollama endpoints are blocked by `core/privacy_guard.py`.
- Spreadsheet rows are not sent to the local model, except a bounded, name-safe sample of an already-redacted result (student names + roster fields; no IDs, contact, financial, or notes) handed to the optional conversational narrator so it can name students on request. Strict Privacy Mode disables that narrator and sends no rows at all.
- The local model cannot edit workbooks directly.
- All commands still pass through validation before execution.
- Edited exports create timestamped local backups.
- Row-removal actions are blocked by default.
- Audit logs store metadata only, not spreadsheet rows.
- Workbook health checks warn before editing sheets with complex formatting.

## Recommended Workstation Controls

For maximum assurance, run the app on a university-managed machine with:

- Wi-Fi and Ethernet disabled while processing sensitive files, when operationally possible.
- macOS firewall enabled.
- Ollama bound to localhost only.
- No VPN or remote desktop session active while handling restricted data.
- Workbook files stored on encrypted local storage.
- `outputs/`, `database/`, and `exports/` stored only on approved local or encrypted drives.
- Browser access limited to the local Streamlit URL.

## Ollama Localhost Binding

Do not expose Ollama to the network.

Use:

```bash
export OLLAMA_HOST=127.0.0.1:11434
ollama serve
```

Do not use:

```bash
export OLLAMA_HOST=0.0.0.0:11434
```

## Optional Network Blocking

For the strongest local-only posture, block outbound network access for the Python/Streamlit process at the operating-system firewall level. The app itself only contains one network path, and it is guarded to localhost Ollama.

## Files That May Contain Local Metadata

- `database/local_learning.db`
- `config/privacy_settings.json`
- `knowledge/learned_synonyms.json`
- `knowledge/learned_column_mappings.json`
- `exports/local_learning_pack.json`
- `outputs/*.xlsx`

These files should be treated as local institutional records and kept on approved storage.

## Role Controls

- Viewer: preview and parse only.
- Editor: execute safe actions and export edited workbooks.
- Admin: manage users, privacy settings, learning packs, local LLM settings, and exports.

Create the first Admin account on initial launch. Do not share Admin credentials.

## Messy Workbook Handling

The app runs a workbook health check before actions. It detects hidden sheets, protected sheets, merged cells, formulas, tables, filters, charts, likely headers, duplicate headers, blank rows, and blank columns.

If a target sheet has complex formatting, the user must explicitly confirm before export. This reduces accidental edits to protected, formula-heavy, or highly formatted sheets.
