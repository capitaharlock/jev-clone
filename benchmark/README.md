# Product comparison benchmark

Two frozen files of nutrition comparison questions built from Open Food Facts by
`asimovia/data/off_products.py`. Every product in them is on the evaluation side of the barcode split, so
none of them appears in any training row.

| File | Rows | sha256 |
|---|---:|---|
| `product_specs.dev.jsonl` | 3,000 | `379c5bfb50cd7c13d04c6f69cccaab1bdb2995407462ff18ed54e15700871a3f` |
| `product_specs.validation.jsonl` | 2,162 | `c1d06411c781b798849c4bb5b6a05357645adc65f4fb8e162cb025315769acee` |

`product_specs.dev.jsonl` is the benchmark. Its rows were sampled evenly over food groups from the
evaluation pool, with a seed shared with other task families of a larger, unpublished suite, which
is why the file is shipped rather than regenerated. The champion checkpoint was chosen on it.

`product_specs.validation.jsonl` holds the rest of the evaluation pool after the sealed quarter
is removed. No training or selection decision ever read it, so its score is the unbiased one. It
was used once, to confirm the slot widths taken from training statistics.

## Row format

```json
{
  "id": "v3-prod-0123456789abcdef",
  "state": "Comparing yogurts on the shelf. Nutrition values are per 100 g as printed on each pack.",
  "question": "Among the products with at most 3.2 g of protein per 100 g, which has the lowest fat?",
  "candidates": [{"id": "c0", "text": "Brand · Name (125 g). energy 61 kcal, fat 1.5 g, ..."}],
  "answer": "c2",
  "group_id": "off/<sorted barcodes>",
  "lang": "en",
  "domain": "food_yogurts",
  "difficulty": "constrained+sodium_mg",
  "decidable": true,
  "license": "ODbL-1.0",
  "source_id": "openfoodfacts"
}
```

- `difficulty` is `extreme` or `constrained`, plus `+sodium_mg` when some cards print sodium in mg.
- `decidable` is false on 25 benchmark rows where a product sits on the cap only after rounding,
  so the printed text no longer decides the answer. They are kept and flagged; results report
  accuracy on all rows and on decidable rows.
- One benchmark id appeared twice in the source pool for two different rows; the second carries
  the suffix `-2`.

## Composition (benchmark)

| | Rows |
|---|---:|
| English / Spanish | 1,938 / 1,062 |
| 3 / 4 / 5 products | 814 / 1,506 / 680 |
| extreme / constrained | 2,160 / 840 |
| with sodium in mg | 240 |

## License

Derived from the Open Food Facts database, available under the
[Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/). These files are
released under the same license. Product data: Open Food Facts contributors,
<https://world.openfoodfacts.org>.
