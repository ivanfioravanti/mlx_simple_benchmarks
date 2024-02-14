import time
import mlx.core as mx
import mlx.nn as nn
from prettytable import PrettyTable, MARKDOWN


def timeit(fun, x):
    # warm up
    for _ in range(10):
        mx.eval(fun(x))
    tic = time.perf_counter()
    for _ in range(100):
        mx.eval(fun(x))
    toc = time.perf_counter()
    return 1e3 * (toc - tic) / 100


def evaluate_improvement(fun_name, fun, t, shape):
    x = mx.random.uniform(shape=shape)
    standard = timeit(fun, x)
    t.add_row([fun_name, standard, "-"])
    compiled = timeit(mx.compile(fun), x)
    percentage_improvement = ((standard - compiled) / standard) * 100
    t.add_row([f"compiled({fun_name})", compiled, f"{percentage_improvement:00.1f}%"])
    return t


table = PrettyTable(["Function", "Duration", "Improvement"])
table.align["Function"] = "l"
table.align["Duration"] = "r"
table.align["Improvement"] = "r"
table.float_format = ".3"
table.set_style(MARKDOWN)

shape = (64, 128, 1024)
table = evaluate_improvement("mx.softmax", mx.softmax, table, shape)
table = evaluate_improvement("nn.gelu", nn.gelu, table, shape)
table = evaluate_improvement("nn.gelu_approx", nn.gelu_approx, table, shape)
table = evaluate_improvement("nn.gelu_fast_approx", nn.gelu_fast_approx, table, shape)
table = evaluate_improvement("nn.relu6", nn.relu6, table, shape)
table = evaluate_improvement("nn.leaky_relu", nn.leaky_relu, table, shape)
table = evaluate_improvement("nn.glu", nn.glu, table, shape)
table = evaluate_improvement("nn.softplus", nn.softplus, table, shape)
table = evaluate_improvement("nn.log_sigmoid", nn.log_sigmoid, table, shape)

print(table)
