# Hyakki Patch Workbench

This workbench stays outside the OAS project so collection data, labels,
training runs, generated models, and virtual environments do not dirty the OAS
workspace.

## Requirements

- OnmyojiAutoScript is installed and runnable.
- The OAS folder contains `toolkit\python.exe`, `module\config`, and
  `tasks\Hyakkiyakou`.
- At least one OAS config exists in `OAS folder\config\*.json`.

The OAS config name does not have to be `oas1`. The workbench lists existing
OAS configs in the collection panel, and also allows typing a custom config
name manually.

## Start

If the workbench and OAS folders are siblings, this is enough. The OAS folder
name does not matter; the launcher identifies it from its internal structure.

```powershell
cd path\to\hyakki-patch-workbench
.\start_collector.ps1
```

If OAS is installed somewhere else, pass the OAS path:

```powershell
.\start_collector.ps1 -OasRoot "D:\path\to\your-oas-folder"
```

Or create `config.local.json` from `config.example.json`:

```json
{
  "oas_root": "D:\\path\\to\\your-oas-folder",
  "config_name": ""
}
```

`config.local.json` is intentionally ignored by git because it contains local
paths.

Open:

```text
http://127.0.0.1:8787/
```

## Training Environment

The workbench can create its own training virtual environment at:

```text
.venv-yolo\
```

It uses OAS `toolkit\python.exe` only to create that environment. Training
packages are installed into `.venv-yolo`, not into OAS itself.
