# Track A: a speech recognizer from scratch

An independent study, not part of the product. The question: **how far does training an ASR model from scratch get on
a small amount of speech, compared with a pretrained model?** The answer is the reason Amanuensis fine-tunes Whisper
instead of training its own model.

Everything here is written by hand where the point is to learn it: log-mel features (`features.py`), CTC greedy and
prefix-beam-search decoding (`decode.py`), the training loop (`train.py`). The model is a small convolutional front end
followed by a bidirectional GRU (or a transformer encoder) trained with CTC loss on characters.

## Run it

```
uv run python -m trackA.fetch dev-clean data/librispeech
uv run python -m trackA.fetch train-clean-100 data/librispeech --max-files 7000     # about 25 h, streamed, stops early
uv run python -m trackA.train --train data/librispeech/LibriSpeech/train-clean-100 \
      --dev data/librispeech/LibriSpeech/dev-clean --hours 10 --epochs 30 --out models/trackA/h10-gru
uv run python -m trackA.compare_whisper                                              # Whisper on the same dev utterances
uv run python -m trackA.evaluate models/trackA/h10-gru/best.pt --whisper REPORT.json  # owner's English eval slice
```

Subsets are nested (the 10 h set is the first 10 h of the 25 h set), so differences come from data, not from a
different draw. Scores are word error rate on 300 LibriSpeech `dev-clean` utterances spread evenly over speakers,
greedy decoding, lowercase letters/space/apostrophe only, no language model.

## Results (RTX 4060 laptop, 30 epochs each, one run per row)

| Model | Training data | Params | Dev WER |
|---|---|---|---|
| from scratch, transformer | 10 h | 3.3 M | 100.0% (never left the CTC plateau) |
| from scratch, BiGRU | 10 h | 1.4 M | 72.8% greedy, 71.7% beam 8 |
| from scratch, BiGRU | 25 h | 1.4 M | 56.6% |
| Whisper tiny (pretrained) | not trained here | 39 M | 8.2% |
| Whisper base (pretrained) | not trained here | 74 M | 5.8% |
| Whisper small (pretrained) | not trained here | 244 M | 3.7% |

Not run: 50 h and 100 h (the plan's full curve). They need the rest of the 6.3 GB corpus and several hours of GPU each:
`--max-files 28539` and `--hours 50` / `--hours 100`.

## What it shows

* Ten hours of speech is not enough to train a usable recognizer from scratch, while an off-the-shelf Whisper model
  (trained on hundreds of thousands of hours) is far better out of the box, before any fine-tuning.
* CTC has a well-known plateau at the start of training, where the model only predicts character frequencies. The
  transformer encoder was still at a training loss of 2.1 after all 30 epochs (about 2.5 for most of the run); the
  recurrent encoder was below 2 by epoch 6 and reached 0.87. Same data, same loss, same schedule: the architecture's
  inductive bias matters when data is scarce.
* More data helps steadily but slowly: 2.5x the data (10 h to 25 h) cut WER from 72.8% to 56.6%. Whisper small is at
  3.7% on the same utterances, so closing the gap by adding data alone would take orders of magnitude more speech.
* Beam search without a language model helps only slightly (about one point).

Caveats: one seed per row, a small model, a fixed epoch budget, no tuning, and no language model. These numbers show
the size of the gap, not the best a from-scratch system could ever do.
