# GLUE & SuperGLUE Benchmarks

Complete evaluation suite for NLP models with GLUE and SuperGLUE tasks.

## GLUE Tasks (9 total)

| Task | Type | Metric | Description |
|------|------|--------|-------------|
| **CoLA** | Single Sentence | Matthews Corr | Linguistic Acceptability |
| **SST-2** | Single Sentence | Accuracy | Sentiment Analysis |
| **MRPC** | Sentence Pair | F1/Acc | Paraphrase Detection |
| **STS-B** | Sentence Pair | Pearson/Spearman | Semantic Similarity (regression) |
| **QQP** | Sentence Pair | F1/Acc | Quora Question Pairs |
| **MNLI** | Sentence Pair | Accuracy | Natural Language Inference |
| **QNLI** | Sentence Pair | Accuracy | Question NLI |
| **RTE** | Sentence Pair | Accuracy | Textual Entailment |
| **WNLI** | Sentence Pair | Accuracy | Winograd NLI |

## Usage

### Download Data

```bash
# Download GLUE data
wget https://gluebenchmark.com/tasks
python download_glue_data.py --data_dir ./data/glue
```

### Run Benchmarks

```bash
# Single task
python benchmarks/run_benchmarks.py \
    --model ./checkpoints/tfan_model.pt \
    --tasks cola

# Multiple tasks
python benchmarks/run_benchmarks.py \
    --model ./checkpoints/tfan_model.pt \
    --tasks cola,sst-2,mrpc,sts-b

# All GLUE tasks
python benchmarks/run_benchmarks.py \
    --model ./checkpoints/tfan_model.pt \
    --suite glue \
    --all
```

### Results

Results saved to `results/benchmark_results.json`:

```json
{
  "cola": {
    "task": "cola",
    "metrics": {"mcc": 0.612},
    "num_examples": 1043
  },
  "sst-2": {
    "task": "sst-2",
    "metrics": {"acc": 0.934},
    "num_examples": 872
  },
  "glue_score": 82.3
}
```

## SuperGLUE Tasks (8 total)

Coming soon!

- **BoolQ** - Boolean Questions
- **CB** - CommitmentBank
- **COPA** - Choice of Plausible Alternatives
- **MultiRC** - Multi-Sentence Reading Comprehension
- **ReCoRD** - Reading Comprehension with Commonsense Reasoning
- **RTE** - Recognizing Textual Entailment
- **WiC** - Word-in-Context
- **WSC** - Winograd Schema Challenge

## Expected Performance

**GLUE Baselines:**
- BERT-Base: 79.6
- BERT-Large: 82.1
- RoBERTa-Base: 83.2
- TFAN (target): **85+**

## Implementation

All tasks implemented in `benchmarks/glue.py`:
- Data processors for each task
- Metrics computation
- Dataset loaders
- Tokenization
