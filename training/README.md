# `training/`

**Owner:** Evidence & Review, with Orchestration & API for packaging

We do not call a hosted language model. We train our own small model to write
review comments, and this folder is how.

## Why train rather than call an API

Two reasons, and both are worth saying in the presentation.

**Independence.** No API key, no external service, no per-request cost, and no
statement data leaving our own infrastructure. In a financial review context
that last point is a compliance property, not a convenience.

**Honesty about what the model does.** Our deterministic engine already knows
what is wrong with a statement. The model's only job is to say it in
professional language. A 250M-parameter model is enough for that, because it is
never asked to reason about numbers - only to express a conclusion already
established as true.

That technique has a name: **knowledge distillation from a rule-based teacher.**
The rules supply the supervision; the model learns the phrasing.

## Files

| File | What it does |
|:--|:--|
| `generate_dataset.py` | Builds training pairs from our own finding logic |
| `finetune_reviewer.ipynb` | Colab notebook - fine-tunes `flan-t5-base` on a T4 GPU |
| `data/` | Generated JSONL. Regenerate rather than commit large versions. |

## Steps

**1. Generate the data**

```bash
python -m training.generate_dataset --n 1200
```

Writes `training/data/train.jsonl` and `validation.jsonl`. Every example pairs a
structured description of a computed finding with the comment a reviewer would
write about it. Figures, companies, severities and phrasings are all randomised,
so the model learns the mapping rather than memorising sentences.

Roughly one example in eight is a **clean** statement with nothing wrong. Without
those the model learns that every input implies a problem, and starts inventing
findings that are not there.

**2. Train**

Open `finetune_reviewer.ipynb` in Google Colab, set the runtime to a T4 GPU,
upload the two JSONL files, and run it top to bottom. Around 15-25 minutes.

**3. Bring the model back**

Download `auditlens-reviewer-final.zip` and unzip it into `models/reviewer/`.

**Do not commit the weights** - they are about 1 GB. `.gitignore` excludes
`models/`. Put them in S3 and let the Docker build pull them in, or attach them
to a GitHub release.

## What to record while training

The presentation needs evidence that a model was trained, not downloaded:

- Base model and parameter count
- Number of training and validation examples
- Final training and validation loss, and the **loss curve image**
- ROUGE-1 and ROUGE-L on the validation set
- **Groundedness rate** - the share of generated comments in which every number
  also appears in the input
- One **before / after** example, showing what the untrained model wrote and what
  the trained one writes

The groundedness figure is the important one. It is our central claim - that the
model never invents a number - expressed as a measurement rather than a promise.

## An honest limitation, and how to answer it

A panel may ask: *if the targets came from your own templates, has the model
learned anything beyond those templates?*

The honest answer is that it has learned to generalise the mapping to unseen
companies, figures and combinations, which is why the validation set uses
examples it never saw in training. It has not learned financial judgement, and
we do not claim it has. The judgement lives in the deterministic engine, on
purpose.
