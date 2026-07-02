# Hyakki Patch Workbench

This directory is intentionally outside the OAS project so collection, labels,
training runs, and generated datasets do not dirty the OAS workspace.

Start the collector:

```powershell
cd D:\OnmyojiAutoScript-easy-install\hyakki-patch-workbench
.\start_collector.ps1
```

Open:

```text
http://127.0.0.1:8787/
```

The collector uses the adjacent OAS install for screenshots and legacy
`oashya` detection through:

```text
D:\OnmyojiAutoScript-easy-install\OnmyojiAutoScript-easy-install
```
