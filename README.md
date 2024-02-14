# mlx_simple_benchmarks
Very simple benchmarks around Apple mlx to compare standard version versus mx.compile.
Adding methods over time.

Here current results on M3 Max (cores: 4E+12P+40GPU)

| Function                      | Duration | Improvement |
|:------------------------------|---------:|------------:|
| mx.softmax                    |    3.120 |           - |
| compiled(mx.softmax)          |    2.332 |       25.3% |
| nn.gelu                       |    4.292 |           - |
| compiled(nn.gelu)             |    0.931 |       78.3% |
| nn.gelu_approx                |    6.121 |           - |
| compiled(nn.gelu_approx)      |    0.940 |       84.6% |
| nn.gelu_fast_approx           |    2.803 |           - |
| compiled(nn.gelu_fast_approx) |    0.946 |       66.3% |
| nn.relu6                      |    1.680 |           - |
| compiled(nn.relu6)            |    0.929 |       44.7% |
| nn.leaky_relu                 |    2.010 |           - |
| compiled(nn.leaky_relu)       |    0.921 |       54.2% |
| nn.glu                        |    1.516 |           - |
| compiled(nn.glu)              |    0.711 |       53.1% |
| nn.softplus                   |    1.028 |           - |
| compiled(nn.softplus)         |    0.923 |       10.2% |
| nn.log_sigmoid                |    2.569 |           - |
| compiled(nn.log_sigmoid)      |    0.923 |       64.1% |

There is a complete builtin benchmark suite in Apple MLX [here](https://github.com/ml-explore/mlx/blob/main/benchmarks/python/comparative/README.md)
I will add compiled version there with a PR.
