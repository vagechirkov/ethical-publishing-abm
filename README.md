# Ethical Publishing ABM

## Installation ([uv](https://docs.astral.sh/uv/getting-started/installation/) should be installed in the system)

```bash
# Create a virtual environment with Python 3.12
uv venv --python 3.12

# Activate the environment (on macOS/Linux)
source .venv/bin/activate

# Install all dependencies from the requirements file
uv pip install -r requirements.txt
```

Test installation
```bash
python3 model.py
```

## Launch Jupyter Notebook

```bash
# Install the Jupyter kernel spec to make it available in notebooks
python -m ipykernel install --user --name ethical_publishing_abm \
       --display-name "Ethical Publishing ABM"

# Optional: To launch the notebook server
jupyter notebook
```