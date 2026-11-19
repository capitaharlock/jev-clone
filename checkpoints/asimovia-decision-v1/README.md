---
license: apache-2.0
library_name: pytorch
pipeline_tag: text-classification
tags: [decision-model, pointer-network, byte-level, from-scratch, curriculum-learning, open-food-facts]
language: [en, es]
datasets: [openfoodfacts]
---

# asimovia-decision-v1

A 16,959,746-parameter byte-level encoder with a pointer head, trained from random weights. Given
a situation, a question and K free-text options, it returns one probability per option. It was
trained for a single skill: comparing food products by their printed nutrition panels, in English
and Spanish, with an optional cap on a second nutrient.

Code, data builder, benchmark and documentation:
<https://github.com/capitaharlock/jev-clone>.

## Scores

| Split | Rows | Forced accuracy | Wilson 95% lower |
|---|---:|---:|---:|
| Benchmark (`product_specs.dev.jsonl`, used to pick the checkpoint) | 3,000 | 96.9% | 96.2% |
| Validation (never used for selection) | 2,162 | 96.2% | 95.3% |

Chance is 26.1%. A rule program that knows the question templates scores 99.2% on the benchmark
(100% on its decidable rows). Latency per decision, one request at a time: 22 ms on an Apple M5
Pro GPU, 67 ms on its CPU.

## Use

```python
from asimovia import load_model, predict  # pip install -e . from the GitHub repository

model = load_model("checkpoints/asimovia-decision-v1")
row = {
    "state": "Comparing soups on the shelf. Nutrition values are per 100 g as printed on each pack.",
    "question": "Which product has the least salt per 100 g?",
    "candidates": [
        {
            "id": "a",
            "text": "Hearth · Tomato Soup. energy 40 kcal, fat 1 g, of which saturates 0.2 g, "
            "sugars 4.1 g, fibre 0.8 g, protein 0.9 g, salt 0.9 g.",
        },
        {
            "id": "b",
            "text": "Hearth · Lentil Soup. energy 60 kcal, fat 0.5 g, of which saturates 0.1 g, "
            "sugars 1.2 g, fibre 2.3 g, protein 3.6 g, salt 0.4 g.",
        },
    ],
}
print(predict(model, [row])[0])  # {'choice': 'b', 'probabilities': {'a': 0.0, 'b': 1.0}}
```

Cards must follow the training layout (`Brand · Name.` then energy, fat, of which saturates,
sugars, fibre, protein, salt); shorter cards can flip the decision.

## Files

| File | What |
|---|---|
| `model.safetensors` | weights, sha256 `00c0fd3bf3c7a3b19f586edf35406e0befde419ed63c49e071abe8331c7799dd` |
| `config.json` | `ModelConfig`: d 384, 6 layers, 8 heads, FFN 1,536, 2 pointer layers, context 1,536 bytes |
| `train.json` | curriculum and selection record |

## Training

2,404 curriculum cycles over 643 hours on one Apple M5 Pro GPU, from random initialization: five
synthetic rungs, then real episodes built from Open Food Facts with labels computed from the
printed numbers. No pretrained checkpoint, teacher model or annotator labels were used.

## Limits

Narrow by design: no knowledge of price, taste, brands, sustainability or other product kinds;
other wording degrades it. The training set carried 1.3% labels that rounding made undecidable
from the text. Spanish trails English by 1.4 points.

## License

Weights and code: Apache-2.0. Training data derived from Open Food Facts (ODbL 1.0, Open Food
Facts contributors).
