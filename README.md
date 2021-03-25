# GraphSAGE
GraphSAGE + Control variate for GCN implemented in PyTorch.

## Usage

All use the PPI dataset by default.

Run the desired file as,
```bash
# Supervised graphsage
python graphsage.py

# Unsupervised graphsage
python graphsage_unsup.py

# Control variate on supervised graphsage
python cv.py

# Control variate on unsupervised graphsage
python cv.py --unsup
```
