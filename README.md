# Consumer Complaint Routing with Transformers

[![CI](https://github.com/dvskaff/complaint-routing-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/dvskaff/complaint-routing-nlp/actions/workflows/ci.yml)
[![Pipeline](https://github.com/dvskaff/complaint-routing-nlp/actions/workflows/pipeline.yml/badge.svg)](https://github.com/dvskaff/complaint-routing-nlp/actions/workflows/pipeline.yml)
![Python](https://img.shields.io/badge/python-3.13-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.15-ee4c2c)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow) ![ONNX](https://img.shields.io/badge/ONNX-int8-lightgrey)

**Live dashboard and in-browser demo: https://dvskaff.github.io/complaint-routing-nlp/** · [Model card](MODEL_CARD.md)

A financial company receives thousands of free-text complaints a day. Can a model read each one and send it to the
right product team, handling the easy cases on its own and passing the uncertain ones to a person? And which issues
are growing? This project answers both on real consumer complaint narratives published by the U.S. Consumer Financial
Protection Bureau (CFPB).

* **Routing:** four models on the same time-based split, from a majority-class floor to TF-IDF + logistic
  regression, frozen sentence embeddings (MiniLM via `sentence-transformers`) + logistic regression, and the same
  MiniLM encoder **fine-tuned end to end** in PyTorch. Scored on the two newest months in their real product mix
  (macro-F1, accuracy, top-2, calibration error).
* **Auto-route vs. human review:** probabilities are calibrated (prior correction + temperature scaling) on a
  validation month; complaints above a confidence threshold that was 95% accurate on validation are routed
  automatically, the rest go to a person with the model's top two suggestions.
* **Emerging issues:** BERTopic (UMAP + HDBSCAN + class-based TF-IDF) on the sentence embeddings of a natural monthly
  sample flags topics whose share of complaints is growing.
* **Serving without a server:** the fine-tuned model is exported to ONNX with the calibration inside the graph,
  quantized to 8-bit, checked against PyTorch, and run **in the browser** with transformers.js and ONNX Runtime
  WebAssembly. Paste a complaint on the dashboard and it is routed on your device.
* **Production habits:** time-based split, hash-based deterministic sampling, cached heavy steps, a quality gate
  (macro-F1 floor, regression check, browser-model parity) before anything is published, a generated model card, and
  tests on synthetic data in CI.

## Results

Rewritten automatically after every pipeline run.

<!-- RESULTS:START -->
_Results appear here after the first run of the **Pipeline** workflow._
<!-- RESULTS:END -->

## The data, and a change in August 2026

The CFPB's Consumer Complaint Database published consumers' own descriptions of their problems (with personal details
redacted) until **August 2026**, when the bureau stopped publishing narratives and moved every narrative it had
published, December 2011 to August 14, 2026, into a downloadable archive in its FOIA reading room. This project reads
those archive files directly. The pipeline checks the archive page monthly; if new files ever appear, the window moves
forward and everything retrains on its own.

Labels are the product the consumer chose when filing (the CFPB's product categories). Training uses the nine months
before the validation month, class-balanced (at most 1,500 complaints per product); validation and test use natural
samples of the newest months, so scores reflect the real mix, which is dominated by credit reporting.

## How it works

```
CFPB narrative archive (zip files) ── src/complaints/data.py: stream CSV chunks, map columns, hash-based samples
   │                                   balanced training set · natural validation / test / topic samples
   ▼
src/complaints/models.py     majority · TF-IDF + LR · MiniLM embeddings + LR · fine-tuned MiniLM (PyTorch)
   │                         calibration: prior correction + temperature (validation month)
   ▼
src/complaints/evaluate.py   accuracy, macro-F1, top-2, ECE · auto-route threshold · confusions · template letters
src/complaints/topics.py     BERTopic on embeddings · monthly topic shares · emerging and fading topics
src/complaints/export.py     ONNX (calibration inside) → int8 → parity check vs. PyTorch
   ▼
quality gate ── dashboard + in-browser demo (GitHub Pages) · README results · MODEL_CARD.md
```

Heavy steps (sample, embeddings, fine-tuned model, topics) are cached in a release asset between runs and recomputed
only when their inputs change.

## Repository layout

```
src/complaints/   config · data · models · evaluate · topics · export · pipeline · report · readme
                  fixture.py: SYNTHETIC archive for tests (pages built from it say so)
scripts/          fetch_web_runtime.sh (transformers.js + ONNX Runtime WebAssembly from npm, served by the site)
tests/            parsing and sampling, models and calibration, routing policy, topics, ONNX export, pipeline, gate
```

## Run it

Everything runs in GitHub Actions. Locally:

```bash
pip install -r requirements-dev.txt
pytest -q                                          # synthetic archive, stand-in models, about 30 seconds
PYTHONPATH=src python -m complaints.pipeline --fixture   # the whole pipeline on synthetic data
PYTHONPATH=src python -m complaints.pipeline             # real data: downloads archive files and MiniLM
```

Data: [CFPB Consumer Complaint Database narratives archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/)
(public). Model: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
Code: MIT license.
