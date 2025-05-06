# GraphPromptor: A Lightweight Reasoning Framework for LLMs on Knowledge Graphs

**GraphPromptor** is a lightweight, efficient reasoning framework based on the "Retrieve → Embed → Reason" methodology described in the paper:

[LightPROF: A Lightweight Reasoning Framework for Large Language Model on Knowledge Graph](https://arxiv.org/abs/2504.03137)

It enables Large Language Models (LLMs), like Gemini 2.5-Flash, to perform complex reasoning over external Knowledge Graphs (KGs) without full model fine-tuning.

By freezing LLM parameters and only training a compact Knowledge Adapter, GraphPromptor delivers high accuracy, efficient memory usage, and faster inference.

**Key features:**

- Multi-hop reasoning with minimal token overhead
- Soft prompt generation by combining graph structure and textual information
- Compatibility with open-source and commercial LLMs
- Flexible knowledge graph backends (NetworkX first, scalable to DGL-KE/PyG)
- Modular design for easy extensions

## Getting Started

This section guides you through setting up and running GraphPromptor.

### Cloning the Repository

Clone the GraphPromptor repository from GitHub:

```bash
git clone https://github.com/AetherForge/GraphPromptor.git
cd GraphPromptor
```

### Installing Dependencies with UV

GraphPromptor uses `uv` for efficient Python package management.

1. **Install uv:** If you don't have `uv` installed, follow the instructions on the [uv documentation](https://docs.astral.sh/uv/tutorial/installation/).

2. **Create a Virtual Environment:** Navigate to the project directory and create a virtual environment:

    ```bash
    uv venv
    ```

3. **Activate the Virtual Environment:**
    - On macOS/Linux:

        ```bash
        source .venv/bin/activate
        ```

    - On Windows:

        ```bash
        .venv\Scripts\activate
        ```

4. **Install Dependencies:** Install the required packages using the `requirements.txt` file:

    ```bash
    uv pip install -r requirements.txt
    ```

    Alternatively, if you modify `requirements.txt`, you can use `uv sync` to install/update dependencies:

    ```bash
    uv sync
    ```

### Setting Environment Variables

Some components, particularly the Reasoning Module, require API keys or other configuration via environment variables.

- **Required Variables:** `GOOGLE_API_KEY` (for accessing Google Gemini models)
- **How to Set:** You can set these variables in your shell or use a `.env` file in the project root directory and a library like `python-dotenv` (included in `requirements.txt`).

    Example `.env` file:

    ```dotenv
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```

    Make sure to replace `"YOUR_API_KEY_HERE"` with your actual API key.

## Usage Example

For a basic usage example, refer to the `notebooks/demo.ipynb` file. This notebook demonstrates the end-to-end workflow from loading a Knowledge Graph to getting an answer from the LLM.

To run the notebook:

1. Ensure you have followed the installation steps above and activated your virtual environment.
2. Install Jupyter if you haven't already (`uv pip install jupyter`).
3. Start the Jupyter notebook server from the project root:

    ```bash
    jupyter notebook
    ```

4. Open `notebooks/demo.ipynb` in your browser.

## Folder Structure

```bash
GraphPromptor/
├── README.md            # Project overview and setup instructions
├── requirements.txt     # Python dependencies managed by uv
├── data/                # Knowledge Graph triples and data files
│   ├── freebase_triples.tsv # Example KG data
│   └── webqsp.jsonl     # Example question data
├── lightprof/           # Core framework modules
│   ├── __init__.py      # Makes lightprof a Python package
│   ├── utils.py         # Utility functions (loaders, tokenizers)
│   ├── retrieval.py     # Retrieval logic (hop prediction, BFS, ranking)
│   ├── adapter.py       # Knowledge Adapter module (embedding fusion)
│   ├── reasoning.py     # Reasoning module (LLM interaction)
│   └── train.py         # Training script for the Knowledge Adapter
├── scripts/             # KG conversion or data preprocessing utilities
└── notebooks/           # Example workflows and demonstrations
    └── demo.ipynb       # Basic usage example notebook
```

## Contribution Guidelines

We welcome contributions! Please see the `docs/overview.md` for contribution guidelines.

## License

(Add license information here, e.g., MIT License)

## Contact

(Add contact information or links here)
