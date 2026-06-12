# qc-streamlit
Collection of streamlit applications for better understanding of quantum chemistry:
  - `hydrogen.py`: Interactive exploration of hydrogen dissociation described by RHF, UHF and CASSCF
  - `orbitals.py`: Visualization of atomic orbitals

## Installation
Clone the repository and install with [`uv`](https://docs.astral.sh/uv/) (uv can be installed with `curl -LsSf https://astral.sh/uv/install.sh | sh` on Linux or macOS):

```bash
git clone https://github.com/jankocivic/qc-streamlit.git
cd qc-streamlit
uv sync
source .venv/bin/activate # Activate virtual environment, should be done every terminal session
```

To run a streamlit application:
```bash
streamlit run <app_name>.py
```
