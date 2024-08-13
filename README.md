# thought
Thought is a light wrapper for llama cpp, the building block for implementing and managing an efficient LLM integration

## Installation

```bash
pip install thought
```

## Get started quickly

```python

include thought.model_downloader
include thought.model_loader

```


## Developing

First create a virtual environment on Python 11 or higher

### Windows

```
python -m venv .venv
.\.venv\Scripts\activate
pip install poetry
poetry install
```

## Build

```
poetry build
python -m twine upload --repository pypi dist/*
```