# LightPROF: A Lightweight Reasoning Framework for LLMs on Knowledge Graphs

**LightPROF** is a lightweight, efficient reasoning framework implementing the methodology described in:

> [LightPROF: A Lightweight Reasoning Framework for Large Language Model on Knowledge Graph](https://arxiv.org/abs/2504.03137)

It enables Large Language Models (LLMs) to perform complex multi-hop reasoning over Knowledge Graphs (KGs) by training only a compact Knowledge Adapter while keeping the LLM frozen.

## Key Features

- **Efficient Training**: Only the Knowledge Adapter is trained (~1M parameters), LLM remains frozen
- **Multi-hop Reasoning**: Supports 1-4 hop reasoning with BERT-based hop prediction
- **Soft Prompt Injection**: Knowledge embeddings injected via `inputs_embeds` for seamless LLM integration
- **Paper-Aligned**: Implements the exact architecture from arXiv:2504.03137
- **Modular Design**: Separate components for retrieval, embedding, and reasoning

## Architecture

```
User Question
      │
      ▼
┌──────────────────┐
│  Hop Predictor   │  ← BERT classifier (1-4 hops)
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
│  │ BERT (frozen)  │ → text_emb       │
│  └────────────────┘                  │
│  ┌────────────────┐                  │
│  │ H+R-T Encoding │ → struct_emb     │
│  └────────────────┘                  │
│  ┌────────────────┐                  │
│  │ KnowledgeEnc   │ → encoded_emb    │
│  └────────────────┘                  │
│  ┌────────────────┐                  │
│  │ Projector MLP  │ → soft_prompts   │
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

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- HuggingFace account with Llama model access

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Mathews-Tom/LightPROF.git
   cd LightPROF
   ```

2. **Create virtual environment:**

   ```bash
   uv venv
   source .venv/bin/activate  # macOS/Linux
   # or: .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**

   ```bash
   uv sync
   ```

4. **Configure environment:**

   Create a `.env` file with your HuggingFace token:

   ```bash
   HF_TOKEN=hf_your_token_here
   ```

   Request access to Llama models at: <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>

## Quick Start

```python
from lightprof.config import LightPROFConfig, ModelConfig, AdapterConfig
from lightprof.adapter import KnowledgeAdapter
from lightprof.reasoning import ReasoningModule
from lightprof.retrieval import RetrievalModule
from lightprof.data import FreebaseKG

# Load configuration
config = LightPROFConfig()

# Load knowledge graph
kg = FreebaseKG("data/freebase_triples.tsv")

# Initialize components
retrieval = RetrievalModule(kg, config.retrieval)
adapter = KnowledgeAdapter(config.model, config.adapter)
reasoning = ReasoningModule(config.model, adapter)

# Run inference
question = "Where was Barack Obama born?"
paths = retrieval.retrieve(question, anchor_entities=["Barack_Obama"])
answer = reasoning.generate(question, paths)
print(answer)
```

## Project Structure

```
LightPROF/
├── lightprof/
│   ├── config.py           # Configuration dataclasses
│   ├── adapter.py          # Knowledge Adapter module
│   ├── retrieval.py        # Retrieval with hop prediction
│   ├── reasoning.py        # LLM reasoning with soft prompts
│   ├── train.py            # Training pipeline
│   ├── models/
│   │   ├── hop_predictor.py      # BERT hop classifier
│   │   ├── knowledge_encoder.py  # Transformer encoder
│   │   └── projector.py          # MLP projector
│   └── data/
│       ├── freebase.py     # FreebaseKG loader
│       ├── webqsp.py       # WebQSP dataset
│       └── cwq.py          # CWQ dataset
├── scripts/
│   ├── evaluate.py         # Hits@1 evaluation
│   └── train_hop_predictor.py
├── tests/                  # Test suite (113 tests)
├── data/                   # Sample data files
└── docs/                   # Documentation
```

## Configuration

LightPROF uses TOML configuration files. Default settings match the paper:

```toml
[model]
llm_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llm_embedding_dim = 4096
bert_model_name = "bert-base-uncased"
freeze_llm = true
freeze_bert = true

[adapter]
structural_mlp_hidden_dim = 64
knowledge_encoder_layers = 2
knowledge_encoder_heads = 8
projector_hidden_dim = 2048

[training]
batch_size = 4
learning_rate = 0.002
num_epochs = 1
```

Load custom configuration:

```python
config = LightPROFConfig.from_toml("configs/webqsp.toml")
```

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=lightprof

# Skip LLM integration tests
SKIP_LLM_TESTS=1 uv run pytest tests/ -v
```

Test results: **113 passed, 5 skipped** (LLM tests require `HF_TOKEN`)

## Training

Train the Knowledge Adapter:

```python
from lightprof.train import Trainer
from lightprof.data import WebQSPDataset, FreebaseKG

# Load data
kg = FreebaseKG("data/freebase.tsv")
train_data = WebQSPDataset("data/webqsp_train.jsonl", kg)

# Initialize trainer
trainer = Trainer(config, kg)

# Train
trainer.train(train_data, num_epochs=1)

# Save checkpoint
trainer.save_checkpoint("checkpoints/adapter.pt")
```

## Evaluation

Evaluate on WebQSP or CWQ:

```bash
uv run python scripts/evaluate.py \
    --config configs/webqsp.toml \
    --checkpoint checkpoints/adapter.pt \
    --dataset webqsp \
    --split test
```

## Paper Results

| Dataset | Hits@1 (Paper) |
|---------|----------------|
| WebQSP  | 83.8%          |
| CWQ     | 59.3%          |

## Citation

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

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome. Please see `docs/overview.md` for guidelines.
