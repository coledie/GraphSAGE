"""
Wrapper that applies control variate, variance reduction
to graph convolutional neural networks.

https://arxiv.org/abs/1710.10568
"""


if __name__ == '__main__':
    from graphsage import GraphSAGE
    model = cv_wrapper(GraphSAGE())
