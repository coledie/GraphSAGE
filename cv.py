"""
Wrapper that applies control variate, variance reduction
to graph convolutional neural networks.

For supervised graphsage,
```bash
python graphsage.py
```

For unsupervised graphsage,
```bash
python cv.py --unsup
```

https://arxiv.org/abs/1710.10568
"""


if __name__ == '__main__':
    from graphsage import GraphSAGE

    model = cv_wrapper(GraphSAGE())

    # add parser arg --unsup for unsueprvised graphsage
