import time
import mlx.core as mx


def calculate_time(fun, x):
    # warm up
    for _ in range(10):
        mx.eval(fun(x))

    tic = time.perf_counter()
    for _ in range(100):
        mx.eval(fun(x))
    toc = time.perf_counter()

    return 1e3 * (toc - tic) / 100


def timeit(fun, x):
    iteration_time = calculate_time(fun, x)
    print(f"Time per iteration {iteration_time:.3f} (ms)")
    return iteration_time


x = mx.random.uniform(shape=(32, 1000, 4096))
standard = timeit(mx.softmax, x)
compiled = timeit(mx.compile(mx.softmax), x)
percentage_improvement = ((standard - compiled) / standard) * 100
