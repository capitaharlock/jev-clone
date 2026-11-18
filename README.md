# asimovia-decision-v1

A 17-million-parameter decision model trained from random weights on one laptop. It reads a
situation, a question and a set of options as raw UTF-8 bytes, and returns one probability per
option. The experiment taught it a single skill, comparing real food products by their printed
nutrition panels in English and Spanish, and it answers **96.9%** of a 3,000-question benchmark
(chance is 26.1%) in about 22 ms per decision.

This is a research artifact, not a product. It shows that a very small model with no pretraining
can learn to read and compare structured text when the data and the curriculum are right. Turning
it into a general decision model would take orders of magnitude more data generation and compute
than this project had: everything here ran on a single Apple M5 Pro.

**[How it works, with diagrams](docs/index.html)** · [Benchmark results](bench/RESULTS.md) ·
[Weights on Hugging Face](https://huggingface.co/capitaharlock/asimovia-decision-v1)

![Exam accuracy per curriculum cycle](docs/assets/curriculum.svg)

## Results

Forced accuracy on the frozen benchmark: the model must pick one of the offered products.

| System | What it is | Rows | Accuracy | p50 latency |
|---|---|---:|---:|---:|
| Rule program | hand-written parser of these exact templates (the ceiling) | 3,000 | 99.2% | 0.05 ms |
| **asimovia-decision-v1** | **this repository, 17 M parameters, random init** | 3,000 | **96.9%** | 22 ms (M5 Pro GPU), 67 ms (CPU) |
| Qwen3.6-35B-A3B | general chat model through Ollama, Q4_K_M, zero-shot, answers one letter | 601 | 91.7% | 589 ms (M5 Pro GPU) |
| TinyJev-0.6B | Qwen3-0.6B-Base plus pointer head, latest revision `c559c2f` | 3,000 | 36.1% | 36 ms (M5 Pro GPU) |
| Uniform guess | | | 26.1% | |

How to read this table:

- **The rule program is the ceiling, not a rival.** It knows the question templates and solves
  every row whose answer the text decides. 25 benchmark rows (0.8%) are not decidable from the
  text; see [the data section](#data).
- **Qwen ran on a 601-row stratified subset** (every cell of difficulty, option count and
  language kept in proportion) because it needs about a second per decision. On those same 601
  rows asimovia-decision-v1 scores 96.3%. Qwen3.6 is the April 2026 generation; Qwen3.8-27B is the
  current open release and could not be downloaded here (see below). Most of Qwen's misses are unit
  conversion: 94.4% when the question asks for salt in grams, 60.4% when it asks for sodium in mg
  and the cards print salt.
- **The specialist beats the generalists on its own task, and only there.** TinyJev and Qwen are
  general models that were never trained on nutrition panels; asimovia-decision-v1 knows nothing
  else. The comparison measures what training on the task buys a model 35 to 2,000 times
  smaller, not which model is better.
- The benchmark row count, the Wilson bounds, per-shape breakdowns and host details for every
  system are in [`bench/RESULTS.md`](bench/RESULTS.md) and `bench/results/*.json`.
- **Selection bias.** The champion checkpoint was picked as the best training cycle on this same
  benchmark. On 2,162 further evaluation-side rows that never took part in any choice
  (`benchmark/product_specs.validation.jsonl`) it scores **96.2%** (Wilson lower bound 95.3%).

Where it still fails, by question shape (benchmark, 3,000 rows):

| Shape | Accuracy |
|---|---:|
| "Which has the most / least X" | 99.0% |
| Same, with a cap on a second nutrient that excludes the naive winner | 91.3% |
| 3 / 4 / 5 products | 98.5% / 97.1% / 94.3% |
| Salt printed in g / some cards print sodium in mg | 97.4% / 91.2% |
| English / Spanish | 97.4% / 96.0% |

### Systems not measured here

The newest decision models run through Ollama's decision API (`/v1/systemone`, Ollama 0.35 or
newer): TinyJev 4B v2 (Qwen3.5-4B, October 2026), Together AI's Tev1 4B and 0.8B, Ollama's Nimble 9B,
and the newest open Qwen, Qwen3.8-27B (August 2026). The network this was built on blocks the Ollama
registry and Hugging Face downloads, so they are not in the table. `bench/run.py` scores them without changes:

```bash
ollama pull nimble                                  # or tev1, parable/tinyjev, qwen3.8:27b
python -m bench.run ollama-decision --model nimble  # decision models
python -m bench.run ollama-chat --model qwen3.8:27b --limit 600
python -m bench.report
```

For context only, from public boards on a different, general benchmark (not comparable with the
table above, not measured by this project): TypeSafe AI's hosted Jev 1.13.0 scores 63.29 on the
JevBench v1.4.2.2 composite, rank 4 of 91 systems ([board](https://benchmarkheaven.com/jev-models));
the TinyJev 4B v2 model card reports 0.766 accuracy on JevBench's 231 public items.

## How it works

### One decision is one sequence

A request has a `state` (the situation), a `question` and `candidates`, each a free-text option with
an id. The model packs all of it into a single byte sequence, up to 1,536 bytes:

```
state | SEP | question | SEP option 1 | SEP option 2 | ... | SEP option K
```

- **Tokenizer.** UTF-8 bytes, plus four control ids (PAD, BOS, EOS, SEP): a vocabulary of 260
  with nothing fitted and nothing downloaded. Accents and units cost a byte or two, never an
  unknown token.
- **Encoder.** Learned byte and position embeddings, then a 6-layer pre-norm Transformer encoder
  (d = 384, 8 heads, FFN 1,536, GELU) with full bidirectional attention. Because the options sit
  in the same sequence, each one is encoded next to the question and next to its rivals; the
  comparison happens inside the encoder, not after it.
- **Span pooling.** The question and each option are mean-pooled over their own bytes. The state
  bytes stay as a token-level memory.
- **Pointer head.** Two cross blocks: the option vectors attend to the state memory, then to each
  other (set attention). Each option's score is a dot product between its contextual vector and a
  key from its own text, scaled by 1/sqrt(d). One extra learned logit, computed from a pooled
  summary, stands for "unknown".
- **Output.** A softmax over the K offered options gives the probabilities; the argmax is the
  choice.

No weight is shaped by a label count or by K. A request can carry any number of options, and an
option never seen in training is scored like any other. That is the difference from a classifier
head: the label set is part of the input, not of the model.

### A fixed layout, found during publication

Each region of the sequence lives in a slot: the state slot is 256 bytes, the question slot 108
and each option slot 191, and a slot grows only when that row's own text is longer, up to a budget.
Unused slot positions are masked out.

The first version padded each slot to the longest text *in the batch*, so where an option started
depended on its neighbours. Training never noticed, because it always ran in batches, but the
same request scored 92.8% alone and 95.4% inside a batch of 32. A packing with no gaps at all
dropped the trained weights to chance: the model had learned to find options by position. The
fix keeps the layout the model was trained on and makes it a function of the row alone: the slot
widths are the median batch widths seen in training, confirmed on the validation split, not on the
benchmark. A row now gets the same logits alone, in any batch, on CPU or GPU, and the unit tests
assert it.

### Theory, in brief

- **Decisions as pointing.** Framing a decision as a pointer over a variable option set comes from
  pointer networks and from typed decision APIs such as Jev's System One: the model scores the
  options it is given instead of generating text.
- **Programmatic supervision.** Every label is computed by a program from numbers printed in the
  input. Nothing a model or an annotator believes enters the targets, so every error is
  checkable and the benchmark has a known ceiling.
- **Curriculum learning.** Rungs of increasing difficulty, with replay of earlier rungs against
  forgetting. A rung is passed at 90%.
- **Symmetry breaking.** The first curriculum asked "most" and "least" at random from step one. For
  any feature that tracks magnitude, the two halves of the batch pull its gradient in opposite
  directions and cancel out; the run sat at 49% on two options. A first rung that only asks "most"
  breaks that symmetry, and the mixed rung was then passed in seven cycles.
- **Hard-example oversampling.** On the real rung, the families that trailed (capped questions,
  five options, sodium in mg) are seen twice per pass.

## Data

### Real products: Open Food Facts

`data/off_products.py` turns the public [Open Food Facts](https://world.openfoodfacts.org) dump
(ODbL 1.0) into episodes:

1. Keep products sold in Spain, the US or the UK with a complete nutrition panel whose salt and
   sodium agree: 117,003 products.
2. Split by a hash of the barcode, 80% training and 20% evaluation, before any question exists. No
   product appears on both sides.
3. Draw 3 to 5 products of one category and ask for the most or the least of one of seven
   nutrients. 60% of the questions add a cap on a second nutrient that rules out the product that
   would otherwise win. 40% of the salt questions print sodium in mg on some cards, so the model
   has to convert (salt = 2.5 x sodium).
4. Keep a question only if the winner beats the runner-up by 15% and a minimum absolute gap.
5. Hash evaluation groups a second time: the lower quarter is a sealed cut that this repository
   never writes.

While preparing this release, a rule program exposed a flaw in step 3: the cap was checked on the
measured values, but the cards print rounded ones. With "at most 0 g of fibre", a product with
0.04 g prints "fibre 0 g" and looks allowed when it is not. Those rows are not decidable from the
text. The builder now marks them `decidable: false` and drops them after every random draw, so
the rest of the output stays byte-identical. The published checkpoint was trained before the fix,
on a set in which 1,046 of 78,140 rows (1.3%) carried such a label; the benchmark keeps its 25
affected rows, flagged, and reports accuracy on both views.

The rebuilt training set (77,094 rows) is identical, row for row and in order, to the one the
model saw, minus those rows.

### Synthetic rungs

`data/curriculum.py` writes the rungs that lead up to the real data, with the same labels by
program and the same 15% margin:

| Rung | Content | Cycles to pass |
|---|---|---:|
| r0-most | two values of one nutrient, always the highest | 3 |
| r1-pair | two values, highest or lowest | 7 |
| r2-panel | 2 to 4 full nutrition panels, find the asked nutrient | 897 |
| r3-units | 3 to 5 named products, half print sodium in mg | 456 |
| r4-constraints | a cap on a second nutrient excludes the naive pick | 374 |
| r5-real | the Open Food Facts episodes, never "passed" | 667 |

Training ran 2,404 cycles over 643 hours of GPU time, from 20 October to 16 November 2026. The
champion is cycle 2394. Each cycle is one call to `training/train.py`: AdamW at 3e-4, 100 warmup
steps then cosine decay to 10%, gradient clipping at 1.0, batches of 8 to 16 decisions with the
same option count, options shuffled at every step so position carries no answer.

While that ran, a local console kept the two jobs on one screen: the CPU job building the next
batch, and the GPU job training and sitting the exam. One frame, cycle 2403 on 16 November 2026,
is in [Watching the run](docs/monitor.html), with a reading of each number. The scores on that
screen are the ones the loop logged live. The table at the top of this file is the later
measurement of the published checkpoint.

## Quick start

Python 3.10 or newer.

```bash
git clone https://github.com/capitaharlock/jev-clone.git asimovia && cd asimovia
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
make weights                      # downloads model.safetensors from Hugging Face
make test                         # unit tests, CPU, under a minute
make serve                        # http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000> for the examples page: six real benchmark comparisons, the JSON sent to
the model, the probabilities it returns and whether it matched the computed answer. Five of the six
match; the miss (`es-constrained-sodium-mg`, five options, sodium cap) is left in on purpose. Edit the JSON
to try your own products.

From the command line:

```bash
curl -s http://127.0.0.1:8000/v1/decide -H 'content-type: application/json' -d '{
  "state": "Comparing breakfast cereals on the shelf. Nutrition values are per 100 g as printed on each pack.",
  "question": "Which product has the least sugar per 100 g?",
  "candidates": [
    {"id": "a", "text": "Nordic Mill · Oat Crunch. energy 410 kcal, fat 9 g, of which saturates 1.6 g, sugars 21 g, fibre 7 g, protein 10 g, salt 0.6 g."},
    {"id": "b", "text": "Nordic Mill · Bran Flakes. energy 350 kcal, fat 2.5 g, of which saturates 0.5 g, sugars 12 g, fibre 15 g, protein 11 g, salt 0.8 g."},
    {"id": "c", "text": "Sunny Bowl · Choco Puffs. energy 390 kcal, fat 3 g, of which saturates 1.1 g, sugars 35 g, fibre 4 g, protein 7 g, salt 0.5 g."}
  ]}'
```

```json
{"choice": "b", "probabilities": {"a": 0.0, "b": 1.0, "c": 0.0}, "latency_ms": 59.4, "model": "asimovia-decision-v1"}
```

Write the cards the way the training data prints them: `Brand · Name.` and then the panel in the
order energy, fat, of which saturates, sugars, fibre, protein, salt. The same three products
written as bare `Name.` cards without the saturates figure got the wrong answer (it chose c).

The same call from Python:

```python
import torch
from model import load_model, predict

model = load_model("checkpoints/asimovia-decision-v1")
row = {"state": "...", "question": "...", "candidates": [{"id": "a", "text": "..."}, ...]}
print(predict(model, [row], torch.device("cpu"))[0])
```

## Reproduce

```bash
make bench                         # this model and the rule program on the benchmark
python -m bench.run tinyjev --model path/to/TinyJev-0.6B --device mps   # needs pip install tinyjev
make report                        # rewrites bench/RESULTS.md

# Rebuild the training data from an Open Food Facts dump
curl -LO https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz
python -m data.off_products en.openfoodfacts.org.products.csv.gz --out work/off

# Train with the curriculum loop; `touch runs/curriculum/STOP` ends it after the current cycle
python -m training.loop --run runs/curriculum
```

A newer dump holds more products, so a rebuild gives a larger, different training set; the
benchmark stays frozen (`sha256` in [`benchmark/README.md`](benchmark/README.md)).

## Repository

```
model/       backbone.py (tokenizer, packing, encoder), head.py (pointer head),
             checkpoint.py, inference.py
data/        off_products.py (Open Food Facts builder), curriculum.py (synthetic rungs), io.py
training/    train.py (one run), loop.py (the curriculum loop)
serve/       server.py (standard-library HTTP server), static/index.html (examples page)
bench/       run.py (one runner for every system), rules.py, report.py, results/
benchmark/   the frozen 3,000-row benchmark and the 2,162-row validation split
examples/    six sample requests with their computed answers
docs/        index.html (diagrams), monitor.html (the run console), the training record and its plot
tests/       unit tests for every module
checkpoints/ config and training report of the published checkpoint
```

## Limits

- One skill. The model compares nutrition panels with an optional cap. It has no knowledge of
  price, taste, brands, sustainability or any other product kind, and other question wording will
  degrade it.
- It reads the card layout it was trained on. Cards in a shorter layout (no brand, no saturates)
  can flip a decision that is obvious to a person; see [Quick start](#quick-start).
- The task is solvable by a short program. Its value is as a controlled test of learning, not as a
  replacement for that program.
- English and Spanish only, and Spanish trails by 1.4 points.
- The champion was selected on the benchmark; the validation split gives the unbiased number.
- Trained on a set with 1.3% undecidable labels (see [Data](#data)).

## Licenses

Code: Apache-2.0 ([LICENSE](LICENSE)). The benchmark, the validation split and the examples are
derived from Open Food Facts and are available under the
[Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/); product data:
Open Food Facts contributors. The weights are released under Apache-2.0.
