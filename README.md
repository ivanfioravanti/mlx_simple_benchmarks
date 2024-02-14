# mlx_simple_benchmarks
Very simple benchmarks around Apple mlx to compare standard version versus mx.compile.
Adding methods over time.

Here current results on M3 Max (cores: 4E+12P+40GPU)

| Function                 | Duration | Improvement |
|--------------------------|----------|-------------|
| mx.softmax               |   11.719 |           - |
| compiled(mx.softmax)     |    8.746 |       25.4% |
| nn.relu                  |    0.221 |           - |
| compiled(nn.relu)        |    0.214 |        3.0% |
| nn.softplus              |    0.382 |           - |
| compiled(nn.softplus)    |    0.359 |        6.1% |
| nn.log_sigmoid           |    0.730 |           - |
| compiled(nn.log_sigmoid) |    0.362 |       50.4% |
|--------------------------|----------|-------------|

There is a complete builtin benchmark suite in Apple MLX [here](https://github.com/ml-explore/mlx/blob/main/benchmarks/python/comparative/README.md)
I will add compiled version there with a PR.
