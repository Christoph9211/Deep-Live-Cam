# Self-extracting archive

This repository includes a helper script that can bundle the complete project into a
single, self-extracting Python executable. The resulting file can be distributed so
that end users only need to download it and execute it to unpack the code and run the
application.

## Prerequisites

* Python 3.8 or newer on the machine that creates the archive.
* The `zipfile` module is part of the Python standard library, so no extra
  dependencies are required.

## Creating the archive

Run the helper script from the repository root:

```bash
python tools/create_self_extracting.py
```

By default, the script produces `dist/Deep-Live-Cam.sfx.py`. The file is marked as
executable; you can move or rename it as needed. To choose a different output path,
pass the `--output` flag:

```bash
python tools/create_self_extracting.py --output /path/to/Deep-Live-Cam.sfx.py
```

## How the archive works

The generated script bundles the full repository (excluding caches and `.git`
metadata) into a base64-encoded ZIP archive and embeds that data directly inside a
Python stub. When executed:

1. A temporary directory is created.
2. The project files are extracted into that directory.
3. If `run.py` is present it is started automatically using the same Python
   interpreter. Otherwise the script simply reports the extraction path so you can
   run commands manually.

## Running on the target machine

After distributing the archive, the recipient can run it directly with Python:

```bash
python Deep-Live-Cam.sfx.py
```

They will see the location where the files were extracted, and the application will
launch immediately if supported on their system. The temporary extraction directory
is not automatically removed so that the unpacked project can be inspected or reused.
