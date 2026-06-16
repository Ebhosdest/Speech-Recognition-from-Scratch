# Speech Recognition From Scratch (Conformer-CTC)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ebhosdest/Speech-Recognition-from-Scratch/blob/main/Conformer_CTC_from_scratch.ipynb)

A speech recognition system built from scratch in PyTorch. No high level ASR
framework, no pretrained weights, no API. The goal was to understand every
part of a modern speech recognition model by building it, not calling it.

Trained on a free GPU. On the standard LibriSpeech test-clean benchmark it
transcribes about 80% of words correctly (WER 0.20), using a model I wrote
and trained myself.

## What it does

Give it speech, it gives you text. The demo in the notebook transcribes audio
the model never saw during training. It gets full sentences word for word, and
where it slips, the mistake is informative (more on that below).

## How it works

1. **Audio to features.** Each clip becomes an 80 dimensional log-mel
   spectrogram (25 ms window, 10 ms hop, 16 kHz). This is the same front end
   used by Whisper and most production systems.
2. **SpecAugment.** Time and frequency masking applied during training, the
   standard regularizer for supervised ASR.
3. **Encoder.** A Conformer (Gulati et al., 2020): 16 layers, model dimension
   144, 4 attention heads, about 8M parameters. Convolution captures local
   acoustic patterns, self attention captures sentence level context.
4. **Loss and alignment.** CTC (Graves, 2006) with a character vocabulary.
   CTC learns the alignment between audio frames and text on its own, so no
   manual per frame labels are needed.
5. **Decoding.** Greedy CTC decoding.

## Data

LibriSpeech `train-clean-100` (100 hours of read English). Evaluated on the
official `test-clean` split, so the numbers are comparable to published work.

## Results

| System | test-clean WER |
|---|---|
| This model, greedy, 100 hours | about 0.20 |
| Conformer-S paper, 960 hours, relative positions | 0.027 |
| Whisper large, zero shot | about 0.02 |

The gap is the point. It shows exactly what more data, relative positional
encoding, and a language model each buy you.

## One interesting failure

On a test clip the model heard the word "bewildered" and wrote "the wildered".
It did not mishear the sound. It heard it correctly, then split it into the
wrong words, because it learned to listen from raw sound and has no dictionary
telling it which words are real. That single behavior is the difference
between a model that hears and one that understands language, and it is exactly
what a language model would fix.

## Honest simplifications (and why)

- **Absolute positional encoding** instead of the relative encoding in the
  paper. Relative gives a small accuracy gain but adds a lot of attention code
  complexity. Knowing that tradeoff was part of the goal.
- **Greedy decoding** instead of beam search with a language model. The most
  common errors here (real words split or misspelled) are exactly what a
  language model would clean up.
- **100 hours** instead of the paper's 960, so it trains on a free GPU.

## Run it yourself

Open the notebook in Google Colab (free T4 GPU is enough).

```
pip install torch torchaudio jiwer matplotlib numpy
```

The notebook is organized as milestones: visualize the features, overfit a
single batch to validate the wiring, train on `train-clean-100`, evaluate on
`test-clean`, then a demo cell that transcribes held-out audio. Switches at
the top of the training and evaluation cells let you reproduce each stage.
