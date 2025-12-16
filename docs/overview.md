# LightPROF: Technical Overview

## 1. Overview

**LightPROF** is a lightweight reasoning framework implementing the methodology from:

> [LightPROF: A Lightweight Reasoning Framework for Large Language Model on Knowledge Graph](https://arxiv.org/abs/2504.03137)

The framework enables LLaMA-3-8B-Instruct to perform multi-hop reasoning over Knowledge Graphs by training only a compact Knowledge Adapter (~1M parameters) while keeping the LLM frozen.

### Design Principles

- **Efficiency**: Only train the Knowledge Adapter, freeze all other components
- **Paper Alignment**: Implements exact architecture from arXiv:2504.03137
- **Modularity**: Separate components for retrieval, embedding, and reasoning
- **Reproducibility**: Configuration-driven with TOML support

## 2. Architecture

### Three-Stage Pipeline

```
Stage 1: Reasoning Graph Retrieval
  ├── Hop Prediction: BERT classifier predicts hop count (1-4)
  ├── Entity Linking: Match question entities to KG nodes
  └── BFS Retrieval: Constrained traversal from anchor entities

Stage 2: Knowledge Embedding (Knowledge Adapter)
  ├── Text Encoding: Frozen BERT encoder for triple text
  ├── Structural Encoding: H+R-T positional MLP
  ├── Fusion: Concatenate text + structural embeddings
  └── Projection: 2-layer MLP to LLM embedding space

Stage 3: Knowledge Prompts Mixed Reasoning
  ├── Soft Prompts: Knowledge embeddings via inputs_embeds
  ├── Hard Prompts: Text instructions for LLM
  └── Generation: Frozen LLaMA-3-8B-Instruct produces answer
```

### Component Specifications

| Component | Implementation | Trainable |
|-----------|---------------|-----------|
| Text Encoder | BERT-base-uncased (768-dim) | No (frozen) |
| Structural Encoder | H+R-T MLP (64-dim) | Yes |
| Knowledge Encoder | 2-layer Transformer | Yes |
| Projector | 2-layer MLP (832 → 4096) | Yes |
| Hop Predictor | BERT + Linear (4 classes) | Separate training |
| LLM | LLaMA-3-8B-Instruct (4096-dim) | No (frozen) |

### Data Flow Diagram

```
User Question
      │
      ▼
┌──────────────────┐
│  Hop Predictor   │  ← BERT classifier predicts 1-4 hops
└──────────────────┘
      │
      ▼
┌──────────────────┐
│  BFS Retrieval   │  ← Constrained graph traversal
└──────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│  Knowledge Adapter                   │
│  ┌────────────────┐                  │
│  │ BERT (frozen)  │ → text_emb [768] │
│  └────────────────┘                  │
│  ┌────────────────┐                  │
│  │ H+R-T Encoding │ → struct_emb [64]│
│  └────────────────┘                  │
│  ┌────────────────┐                  │
│  │ Concatenate    │ → fused [832]    │
│  └────────────────┘                  │
│  ┌────────────────┐                  │
│  │ KnowledgeEnc   │ → encoded [832]  │
│  └────────────────┘                  │
│  ┌────────────────┐                  │
│  │ Projector MLP  │ → prompts [4096] │
│  └────────────────┘                  │
└──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│  LLaMA-3-8B-Instruct (frozen)        │
│  inputs_embeds = soft + hard prompts │
└──────────────────────────────────────┘
      │
      ▼
   Answer
```

## 3. Module Details

### 3.1 Retrieval Module

**Location**: `lightprof/retrieval.py`

The Retrieval Module extracts relevant reasoning paths from the Knowledge Graph.

**Process**:
1. **Hop Prediction**: BERT-based classifier predicts reasoning depth (1-4 hops)
2. **Entity Linking**: Match question entities to KG nodes using fuzzy matching
3. **BFS Traversal**: Breadth-first search up to predicted hop count
4. **Path Sampling**: Select top-k paths based on relevance

```python
from lightprof.retrieval import RetrievalModule
from lightprof.data import FreebaseKG

kg = FreebaseKG("data/freebase_triples.tsv")
retrieval = RetrievalModule(kg, config.retrieval)

question = "Where was Barack Obama born?"
paths = retrieval.retrieve(question, anchor_entities=["Barack_Obama"])
# Returns: [[("Barack_Obama", "place_of_birth", "Honolulu")]]
```

### 3.2 Knowledge Adapter

**Location**: `lightprof/adapter.py`

The Knowledge Adapter converts reasoning paths into soft prompt embeddings.

**Architecture**:
- **BERT Encoder** (frozen): Encodes triple text → 768-dim
- **Structural Embedding**: H+R-T positional encoding → 64-dim
- **Concatenation**: Fuses text and structure → 832-dim
- **Knowledge Encoder**: 2-layer Transformer → 832-dim
- **Projector**: 2-layer MLP → 4096-dim (LLaMA embedding space)

```python
from lightprof.adapter import KnowledgeAdapter

adapter = KnowledgeAdapter(config.model, config.adapter)
soft_prompts = adapter(paths)  # [num_paths, 4096]
```

**H+R-T Encoding** (Paper Equation 4):
```
E_structural = MLP(E_head + E_relation - E_tail)
```

### 3.3 Reasoning Module

**Location**: `lightprof/reasoning.py`

The Reasoning Module generates answers using LLaMA-3-8B-Instruct with soft prompt injection.

**Soft Prompt Injection**:
```python
# Combine soft prompts with hard prompt embeddings
combined_embeds = torch.cat([soft_prompts, text_embeds], dim=1)

# Generate using inputs_embeds (not input_ids)
outputs = model.generate(
    inputs_embeds=combined_embeds,
    attention_mask=attention_mask,
    max_new_tokens=64,
)
```

**Hard Prompt Template**:
```
Based on the knowledge graphs, answer the following question.
Question: {question}
Answer:
```

### 3.4 Hop Predictor

**Location**: `lightprof/models/hop_predictor.py`

BERT-based classifier for predicting reasoning hop count.

**Architecture**:
- BERT encoder (frozen or fine-tuned)
- Linear classifier: 768 → 4 (1-4 hops)
- CrossEntropyLoss with 1-indexed labels

```python
from lightprof.models import HopPredictor

predictor = HopPredictor(config.model)
hop_count = predictor.predict("Where was Obama born?")  # Returns 1-4
```

## 4. Configuration

### TOML Configuration

```toml
[model]
llm_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llm_embedding_dim = 4096
bert_model_name = "bert-base-uncased"
bert_embedding_dim = 768
freeze_llm = true
freeze_bert = true

[adapter]
structural_mlp_hidden_dim = 64
knowledge_encoder_layers = 2
knowledge_encoder_heads = 8
projector_hidden_dim = 2048
dropout = 0.1

[retrieval]
max_hops = 4
top_k_relations = 10
max_paths_per_question = 20
use_hop_predictor = true

[training]
batch_size = 4
learning_rate = 0.002
num_epochs = 1
warmup_ratio = 0.1
weight_decay = 0.01
```

### Loading Configuration

```python
from lightprof.config import LightPROFConfig

# Default configuration (paper-aligned)
config = LightPROFConfig()

# Load from TOML file
config = LightPROFConfig.from_toml("configs/webqsp.toml")
```

## 5. Data Formats

### Knowledge Graph (TSV)

```
head<TAB>relation<TAB>tail
Barack_Obama	place_of_birth	Honolulu
Honolulu	located_in	Hawaii
Hawaii	country	United_States
```

### QA Dataset (JSONL)

**WebQSP Format**:
```json
{
  "question": "Where was Barack Obama born?",
  "answer": ["Honolulu"],
  "entities": ["Barack_Obama"],
  "hops": 1
}
```

**CWQ Format**:
```json
{
  "question": "What is the capital of the country where Obama was born?",
  "answer": ["Washington, D.C."],
  "entities": ["Barack_Obama"],
  "hops": 3
}
```

## 6. Training

### Training the Knowledge Adapter

```python
from lightprof.train import Trainer
from lightprof.data import WebQSPDataset, FreebaseKG

# Load data
kg = FreebaseKG("data/freebase.tsv")
train_data = WebQSPDataset("data/webqsp_train.jsonl", kg)

# Initialize trainer
trainer = Trainer(config, kg)

# Train (only adapter is updated, LLM frozen)
trainer.train(train_data, num_epochs=1)

# Save checkpoint
trainer.save_checkpoint("checkpoints/adapter.pt")
```

### Training the Hop Predictor

```bash
uv run python scripts/train_hop_predictor.py \
    --train_data data/webqsp_train.jsonl \
    --val_data data/webqsp_val.jsonl \
    --output_dir checkpoints/hop_predictor
```

### Training Configuration (Per Paper)

| Parameter | Value |
|-----------|-------|
| Batch Size | 4 |
| Learning Rate | 2e-3 |
| Optimizer | AdamW |
| Scheduler | Cosine annealing |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.01 |
| Epochs | 1 |

## 7. Evaluation

### Hits@1 Metric

The primary evaluation metric is Hits@1 (exact match accuracy after normalization).

```bash
uv run python scripts/evaluate.py \
    --config configs/webqsp.toml \
    --checkpoint checkpoints/adapter.pt \
    --dataset webqsp \
    --split test
```

### Paper Results

| Dataset | Hits@1 |
|---------|--------|
| WebQSP  | 83.8%  |
| CWQ     | 59.3%  |

## 8. Project Structure

```
LightPROF/
├── lightprof/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration dataclasses
│   ├── adapter.py            # Knowledge Adapter module
│   ├── retrieval.py          # Retrieval with hop prediction
│   ├── reasoning.py          # LLM reasoning with soft prompts
│   ├── train.py              # Training pipeline
│   ├── utils.py              # Utilities
│   ├── models/
│   │   ├── hop_predictor.py  # BERT hop classifier
│   │   ├── knowledge_encoder.py  # Transformer encoder
│   │   └── projector.py      # MLP projector
│   └── data/
│       ├── freebase.py       # FreebaseKG loader
│       ├── webqsp.py         # WebQSP dataset
│       └── cwq.py            # CWQ dataset
├── scripts/
│   ├── evaluate.py           # Hits@1 evaluation
│   └── train_hop_predictor.py
├── tests/                    # Test suite (113 tests)
├── data/                     # Sample data files
└── docs/                     # Documentation
```

## 9. Testing

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=lightprof

# Skip LLM integration tests
SKIP_LLM_TESTS=1 uv run pytest tests/ -v
```

### Test Results

- **113 passed**: All unit and integration tests
- **5 skipped**: LLM integration tests (require `HF_TOKEN` for Llama access)

### Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `HF_TOKEN` | HuggingFace API token for Llama | For LLM tests |
| `SKIP_LLM_TESTS` | Set to "1" to skip LLM tests | Optional |

## 10. Key Implementation Details

### TransformerEncoder Configuration

The Knowledge Encoder uses pre-norm architecture with nested tensors disabled:

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=input_dim,
    nhead=num_heads,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    activation="gelu",
    batch_first=True,
    norm_first=True,  # Pre-norm architecture
)

self.encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=num_layers,
    enable_nested_tensor=False,  # Required with norm_first=True
)
```

### Soft Prompt Injection

Embeddings are injected via `inputs_embeds` parameter:

```python
# Get LLM embedding layer
embed_layer = model.get_input_embeddings()

# Convert hard prompt to embeddings
text_embeds = embed_layer(input_ids)

# Concatenate soft prompts with hard prompt embeddings
combined = torch.cat([soft_prompts.unsqueeze(0), text_embeds], dim=1)

# Generate with combined embeddings
outputs = model.generate(inputs_embeds=combined, ...)
```

### Hop Predictor Labels

Labels are 1-indexed (1-4 for hop counts):

```python
# In training
loss = F.cross_entropy(logits, labels - 1)  # Convert to 0-indexed

# In prediction
predicted_hops = logits.argmax(dim=-1) + 1  # Convert back to 1-indexed
```

## 11. Future Work

1. **Full Training Run**: Train on complete WebQSP and CWQ datasets
2. **Evaluation Pipeline**: Benchmark against paper results
3. **Model Optimization**: Add quantization support for lower memory
4. **Inference Optimization**: Batch processing and caching
5. **Additional Datasets**: Support for MetaQA and other KGQA benchmarks

## 12. Citation

If you use this implementation, please cite both the original paper and this repository:

**Original Paper:**
```bibtex
@misc{ao2025lightproflightweightreasoningframework,
      title={LightPROF: A Lightweight Reasoning Framework for Large Language Model on Knowledge Graph},
      author={Tu Ao and Yanhua Yu and Yuling Wang and Yang Deng and Zirui Guo and Liang Pang and Pinghui Wang and Tat-Seng Chua and Xiao Zhang and Zhen Cai},
      year={2025},
      eprint={2504.03137},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.03137},
}
```

**This Implementation:**
```bibtex
@software{mathews2025lightprof,
      title={LightPROF: PyTorch Implementation of Lightweight Reasoning Framework for LLMs on Knowledge Graphs},
      author={Tom Mathews},
      year={2025},
      url={https://github.com/Mathews-Tom/LightPROF},
      note={Open-source implementation with LLaMA-3-8B-Instruct and BERT-based Knowledge Adapter}
}
```

## 13. References

- Paper: [arXiv:2504.03137](https://arxiv.org/abs/2504.03137)
- LLaMA-3: [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- BERT: [bert-base-uncased](https://huggingface.co/bert-base-uncased)
