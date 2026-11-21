
# Memora: Memory-Augmented Transformers

Memora is an implementation of the Titans memory-augmented transformer architecture, exploring neural memory, segmented attention, and efficient processing of long sequences.

## Overview

Memora extends the standard transformer architecture with a neural memory mechanism that retains information across sequence segments. It combines attention-based context processing with persistent memory to investigate how transformers can handle longer contexts efficiently.

## Features

- **Segmented Attention:** Processes long sequences in smaller segments.
- **Neural Memory:** Maintains and updates persistent memory across sequence segments.
- **Sliding Window Attention:** Focuses attention on local context to reduce computational overhead.
- **Configurable Architecture:** Supports customization of model dimensions, depth, attention heads, segment length, and memory settings.
- **Attention Visualization:** Provides tools to inspect attention patterns, layer behavior, and memory activations.
- **Model Evaluation:** Supports perplexity and loss evaluation, text generation, and comparative experiments.
- **Long-Context Retrieval:** Includes a Needle-in-a-Haystack evaluation for testing information retrieval from extended contexts.

## Installation

```bash
git clone <repository-url>
cd Memora
pip install -r titans/requirements.txt
```

## Usage

### Training

Train a model on WikiText-103 with neural memory and sliding window attention enabled.

```bash
python main.py --mode train \
  --dataset wikitext \
  --wikitext_version 103 \
  --data_path ./data \
  --model_dim 512 \
  --depth 12 \
  --heads 8 \
  --seq_length 1024 \
  --batch_size 4 \
  --val_batch_size 2 \
  --gradient_accumulation 4 \
  --fp16 \
  --output_dir ./outputs/wikitext_model \
  --learning_rate 3e-4 \
  --use_memory \
  --sliding_window
```

For GPU memory constraints, reduce the batch size or sequence length, enable mixed-precision training, or increase gradient accumulation.

### Evaluation

Evaluate a trained checkpoint using the validation dataset.

```bash
python main.py --mode eval \
  --dataset wikitext \
  --wikitext_version 103 \
  --data_path ./data \
  --checkpoint ./outputs/wikitext_model/checkpoints/best_model_10000.pt \
  --val_batch_size 2
```

### Evaluation with Visualizations

Generate evaluation metrics, text samples, and visualizations.

```bash
python3 evaluate_model_with_visualizations.py \
  --checkpoint ./outputs/wikitext_model/checkpoints/best_model_10000.pt \
  --model_dim 512 \
  --depth 12 \
  --heads 8 \
  --use_memory \
  --vis_all \
  --dataset wikitext \
  --data_path ./data \
  --wikitext_version 103 \
  --num_gen_samples 5 \
  --temperature 0.7
```

The evaluation outputs include:

- Validation loss and perplexity
- Generated text samples
- Attention pattern visualizations
- Memory activation and usage visualizations
- Generation animations
- Evaluation reports in JSON and human-readable formats

### Text Generation

Generate text from a trained checkpoint using a prompt.

```bash
python main.py --mode generate \
  --checkpoint ./outputs/wikitext_model/checkpoints/best_model_10000.pt \
  --prompt "Once upon a time" \
  --max_length 100 \
  --temperature 0.8 \
  --top_k 50 \
  --top_p 0.9
```

### Model Comparison

Compare the memory-augmented architecture against a standard transformer.

```bash
python compare_models.py \
  --dim 512 \
  --depth 6 \
  --heads 8 \
  --segment-len 512 \
  --batch-size 16 \
  --steps 5000 \
  --eval-interval 500 \
  --dataset wikitext \
  --output-dir ./outputs/model_comparison
```

The comparison covers training metrics, inference performance, memory usage, and text generation quality.

### Needle-in-a-Haystack Evaluation

Test whether the model can retrieve specific information embedded in a larger context.

```bash
python3 simple_needle_test.py \
  --checkpoint ./outputs/checkpoints/final_562.pt \
  --tokenizer distilgpt2 \
  --model_dim 256 \
  --depth 6 \
  --heads 8 \
  --dim_head 64 \
  --segment_len 512 \
  --max_context_len 512 \
  --needle "Donald Trump is Blonde" \
  --haystack_dir ./haystack \
  --retrieval_question "How does Donald Trump look" \
  --output_dir ./simple_test_results \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 50
```

The test measures the model's ability to retrieve relevant information from a supplied context in response to a query.

## Architecture

Memora combines transformer layers, segmented attention, and a persistent neural memory mechanism.

### Transformer Backbone

The transformer backbone consists of multi-head attention and feedforward layers. It provides contextual representations for input tokens.

### Segmented Attention

Input sequences are processed in segments rather than as a single, unbounded sequence. This allows attention computation to be organized around manageable portions of the input.

### Neural Memory

The memory component maintains information across processing steps and provides relevant stored representations to the model. Its update mechanism uses a surprise signal, with momentum and forgetting mechanisms to regulate memory updates.

### Positional Embeddings

Continuous axial positional embeddings represent token positions across extended sequences.

## Attention and Memory Visualization

Memora includes visualization tools for inspecting the model's internal behavior:

- **Attention Patterns:** Examine token-to-token attention relationships.
- **Attention Head Analysis:** Compare the behavior of individual attention heads.
- **Layer Analysis:** Inspect attention behavior across transformer layers.
- **Memory Activations:** Examine memory usage and update patterns.
- **Generation Animations:** Observe attention patterns during autoregressive generation.

## Project Structure

```text
Memora/
├── main.py
├── compare_models.py
├── evaluate_model_with_visualizations.py
├── simple_needle_test.py
└── titans/
    ├── config.py
    ├── models/
    │   ├── attention.py
    │   ├── memory.py
    │   ├── transformer.py
    │   └── utils.py
    ├── data/
    │   └── dataloader.py
    ├── training/
    │   ├── trainer.py
    │   ├── utils.py
    │   └── metrics.py
    ├── visualizations/
    │   ├── plots.py
    │   ├── attention_viz.py
    │   └── comparison_plots.py
    ├── evaluation/
    └── requirements.txt
```

## Configuration

Model and training parameters can be configured through command-line arguments and `titans/config.py`.

| Parameter | Description |
|---|---|
| `model_dim` | Hidden dimension of the model |
| `depth` | Number of transformer layers |
| `heads` | Number of attention heads |
| `segment_len` | Number of tokens in each segment |
| `use_memory` | Enables the neural memory component |
| `batch_size` | Number of samples processed per training batch |
| `learning_rate` | Optimizer learning rate |
| `warmup_steps` | Number of learning-rate warmup steps |
| `fp16` | Enables mixed-precision training |
| `label_smoothing` | Controls label smoothing during training |

Refer to `titans/config.py` and the command-line argument definitions for the available settings.

## Datasets

Memora supports:

- **WikiText-2:** A language modeling dataset for smaller-scale experiments.
- **WikiText-103:** A larger language modeling dataset for long-context experiments.
- **Custom Text Datasets:** Text data supplied through the project's data-loading utilities.

## Evaluation and Analysis

The evaluation scripts support analysis across several dimensions:

- **Language Modeling:** Validation loss and perplexity.
- **Generation:** Text samples produced under configurable sampling parameters.
- **Efficiency:** Training speed, inference latency, and memory consumption.
- **Memory Behavior:** Memory activations and update patterns.
- **Architecture Comparison:** Comparative evaluation of memory-augmented and standard transformer models.
- **Context Retrieval:** Needle-in-a-Haystack testing for information retrieval.
