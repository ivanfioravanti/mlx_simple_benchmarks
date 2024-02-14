# mlx_simple_benchmarks
Very simple benchmarks around Apple mlx to compare standard version versus mx.compile.
Adding methods over time.

Here current results on M3 Max (cores: 4E+12P+40GPU)

| Function                      | Duration | Improvement |
|:------------------------------|---------:|------------:|
| mx.softmax                    |    0.603 |           - |
| compiled(mx.softmax)          |    0.464 |       23.2% |
| nn.gelu                       |    0.591 |           - |
| compiled(nn.gelu)             |    0.279 |       52.7% |
| nn.gelu_approx                |    0.803 |           - |
| compiled(nn.gelu_approx)      |    0.292 |       63.6% |
| nn.gelu_fast_approx           |    0.436 |           - |
| compiled(nn.gelu_fast_approx) |    0.269 |       38.3% |
| nn.relu6                      |    0.354 |           - |
| compiled(nn.relu6)            |    0.267 |       24.6% |
| nn.leaky_relu                 |    0.358 |           - |
| compiled(nn.leaky_relu)       |    0.267 |       25.5% |
| nn.glu                        |    0.330 |           - |
| compiled(nn.glu)              |    0.234 |       29.2% |
| nn.softplus                   |    0.298 |           - |
| compiled(nn.softplus)         |    0.252 |       15.3% |
| nn.log_sigmoid                |    0.459 |           - |
| compiled(nn.log_sigmoid)      |    0.273 |       40.4% |

There is a complete builtin benchmark suite in Apple MLX [here](https://github.com/ml-explore/mlx/blob/main/benchmarks/python/comparative/README.md)
I will add compiled version there with a PR.
