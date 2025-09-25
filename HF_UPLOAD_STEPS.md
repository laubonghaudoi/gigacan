# HuggingFace Upload Workflow

## 1. Prepare Environment & Metadata

- Ensure `download/metadata.csv` lists every `.opus` file you want to publish with columns: `id`, `audio`, `title`, `description`, `publish_date`, `duration`, `duration_seconds` (expand as needed).
- Activate the project venv and install tooling:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  pip install datasets webdataset
  ```

## 2. Bundle WebDataset Shards

- Create shards once metadata is final. Dry-run first:
  ```bash
  python make_webdataset.py metadata.csv webdataset --dry-run
  ```
- When the plan looks right, run without `--dry-run`. Default shard targets ≈20 files / ≤1.5 GB; adjust with `--samples-per-shard` and `--max-shard-size-mb` if needed.
- Inspect a shard to confirm each entry pairs `legco-<year>-NNNNNN.opus` with a JSON payload containing the metadata fields.

## 3. Local Validation

- Spot-check the dataset structure with the `datasets` loader:
  ```python
  from datasets import load_dataset
  ds = load_dataset("webdataset", data_files={"2025": "webdataset/2025/*.tar"})
  print(ds["2025"][0])
  ```
- Confirm that the `audio` key resolves to the tar member name and other columns mirror the CSV.

## 4. Generate dataset_infos.json

- Produce Hub-compatible schema metadata:
  ```bash
  python generate_dataset_infos.py webdataset --output dataset_infos.json --sampling-rate 16000
  ```
- The script auto-discovers yearly splits under `webdataset/` and casts the `audio` column to `Audio(16000)`. Review the resulting JSON before publishing.

## 5. Prepare Repo Artefacts

- Create a dataset card (README) describing source, licensing, preprocessing, and split definitions.
- Add `.gitattributes` with `*.tar filter=lfs diff=lfs merge=lfs` so shards use Git LFS.
- Optionally copy `download/metadata.csv` to `metadata/segments.csv` for transparency.

## 6. Upload to HuggingFace

- Authenticate and create the dataset repo:
  ```bash
  huggingface-cli login
  huggingface-cli repo create <namespace>/<dataset-name> --type dataset
  ```
- Push artefacts:
  ```bash
  git lfs track "*.tar"
  git add webdataset dataset_infos.json metadata README.md .gitattributes
  git commit -m "Add LegCo WebDataset shards"
  git push
  ```

## 7. Verify on the Hub

- Open the Dataset Viewer for each split (2025, 2024, …). The audio column should render a player, and metadata columns should match the CSV headers.
- Test streaming:
  ```python
  ds = load_dataset("<namespace>/<dataset-name>", streaming=True)
  next(iter(ds["2025"]))
  ```
- Once confirmed, share the dataset card link with collaborators.
