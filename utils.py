"""
********************************************************************************
utility
********************************************************************************
"""

import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import numbers
import pickle as pkl
import yaml

def find_all_words(string):
    _len = len(string)
    list_of_words = []
    start = 0
    while start < _len:
        for end in range(start + 1, _len + 1):
            if end == _len:
                list_of_words.append((start, string[start:end]))
                start = _len
                break
            elif not string[start:end].isidentifier():
                list_of_words.append((start, string[start : end - 1]))
                start = end
                break
    return list_of_words


def replace_words(string, replacement_dict):
    list_of_words = find_all_words(string)
    shift = 0
    for word in list_of_words:
        if word[1] in replacement_dict.keys():
            start = word[0]
            end = start + len(word[1])
            string = (
                string[: start + shift]
                + replacement_dict[word[1]]
                + string[end + shift :]
            )
            shift += len(replacement_dict[word[1]]) - len(word[1])
    return string


def gen_points(num, bounds, n_vars=None):
    n_vars = len(bounds)
    points = [0] * n_vars
    for i in range(n_vars):
        points[i] = tf.random.uniform(
            (int(num), 1), bounds[i][0], bounds[i][1], dtype=tf.float32
        )
    return tf.Variable(tf.concat(points, axis=1))


def gen_condition(cond_dict, model_args, **kwargs):
    def default_xc():
        x = gen_points(cond_dict["N"], cond_dict["point_area"])
        right_side_line = line_parser(cond_dict["right_side"], **kwargs)
        right_side_func = eval(
        "lambda vars: (" + right_side_line + ",)", kwargs | {"tf": tf}
        )
        c = tf.convert_to_tensor(right_side_func(x), dtype=tf.float32)
        return x, c
    try: 
        if cond_dict['raw_data_condition']:
            with open(cond_dict['filename'], mode="rb") as datafile:
                data = pkl.load(datafile)
            x = tf.convert_to_tensor(np.array((data['points'])), dtype=tf.float32)
            c = tf.convert_to_tensor(np.array((data['data'])), dtype=tf.float32)
        else:
            x, c = default_xc()
    except KeyError:
            x, c = default_xc()
    x = x*2/(model_args['in_ub']-model_args['in_lb']) - 1 #normalization
    eq_string = line_parser("( " + cond_dict["eq_string"] + " ,)", **kwargs)
    if "d/d" in cond_dict["eq_string"]:
        compute_grads = True
    else:
        compute_grads = False
    eq_string = compile(eq_string, "<string>", "eval")
    return (x, c, eq_string, compute_grads)

def eval_dict(d, kwargs={}, recursion=0):
    if recursion == 0:
        for key in d.keys():
            if key not in ["eq_string", "act", "right_side", "filename"]:
                if isinstance(d[key], numbers.Number):
                    #d[key] = tf.cast(d[key], tf.float32)
                    pass
                else:
                    d[key] = eval(str(d[key]), kwargs | d)
        return d
    else:
        for key in d.keys():
            if key not in ["eq_string", "act", "right_side"]:
                d[key] = eval_dict(d[key], kwargs | d, recursion - 1)
        return d


default_var_names = ("x", "y")


def line_parser(eq_string, func_names, var_names=default_var_names, **kwargs):
    var_dict = dict(zip(var_names, range(len(var_names))))
    splited = eq_string.split(" ")
    ops_stack = []

    def is_der_operator(string: str):
        if re.findall(r"\(d\/d..?\)", string):
            return True
        else:
            return False

    def apply_ops(ops_stack: list, func: str, var_dict: dict):
        dif_powers = [0] * len(var_dict)
        for op in ops_stack:
            op = op.replace("(d/d", "")
            op = op.replace(")", "")
            op = op.split("^")

            var_index = var_dict[op[0]]
            try:
                power = op[1]
            except IndexError:
                power = 1
            dif_powers[var_index] = int(power)
        # previous = ''
        f_name = "u_"
        dif_string = ""
        dif_index = ""
        # standard u_ has shape (n,m), u_x (n,m,x) u_xx (n,m,x,x) and so on
        for i in range(len(var_dict)):
            dif_string += "x" * dif_powers[i]
            dif_index += ("," + str(i)) * dif_powers[i]
        func_index = func_names.index(func)
        return f_name + dif_string + "[:," + str(func_index) + dif_index + "]"

    res = ""
    for i in range(len(splited)):
        if is_der_operator(splited[i]):
            ops_stack.append(splited[i])
        elif splited[i] in func_names:
            res += apply_ops(ops_stack, splited[i], var_dict)
            ops_stack = []
        elif splited[i] in var_names:
            res += "vars[:," + str(var_dict[splited[i]]) + "]"
        else:
            res += splited[i]
    return res


def make_logger(add_data=None):
    now = datetime.datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M_%S")

    f_path = "./results/"
    f_name = now + ".txt"
    path = os.path.join(f_path, f_name)

    with open(path, mode="a") as f:
        print(add_data, file=f)
    return path


def write_logger(path, log):
    with open(path, mode="a") as f:
        print(log, file=f)


"""
def plotting(func, xlabel, ylabel, title=''):
    def wrapper(*args, **kwargs):
        fig, ax = plt.subplots(figsize=(8, 4))
        func(*args, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.clf()
        plt.close()
    return wrapper

@plotting
def plot_compari(epoch, x, y, u_inf)
"""

def plot_comparison1d(
    epoch,
    x,
    y,
    u_inf,
    xlabel,
    ylabel,
    title="",
):

# fig, ax = plt.subplots(figsize=(4, 4))
    # ax.set_xticks
    xticks = (np.max(x) - np.min(x)) / 4.0
    plt.plot(x, u_inf)  # , vmin=umin, vmax=umax)
    #plt.xticks(np.arange(np.min(x), np.max(x) + 1e-6, (np.max(x)-np.min(x))/5), xticks)
    plt.xticks(np.arange(np.min(x), np.max(x) + 1e-6, xticks))
    plt.xlim(np.min(x), np.max(x))
    plt.xlabel(xlabel)
    #plt.title("inference")

    plt.savefig("./results/comparison_" + title + "_" + str(epoch) + ".pdf", dpi=300)
    plt.clf()
    plt.close()
    with open('./results/data_'+ title + "_" + str(epoch) + ".txt"	, 'wb') as f:
        pkl.dump((x,y,u_inf), f)	
        #pkl.dump(data, f)
        #	pkl.dump(data, f)
        #print(str(x), file=f)
        #print(str(y), file=f)
        #print(str(u_inf), file=f)
       
def load_data(title, epoch):
    with open('./results/data_'+ title + "_" + str(epoch) + ".txt"	, 'b') as f:
        x,y,u_inf = pkl.load(f)
    plot_comparison(epoch,x,y,u_inf,
                        xlabel='',
                        ylabel='',
                        title=title)
    return x,y,u_inf


def plot_comparison(
    epoch,
    x,
    y,
    u_inf,
    xlabel,
    ylabel,
    title="",
):
    # fig, ax = plt.subplots(figsize=(4, 4))
    # ax.set_xticks
    xticks = (np.max(x) - np.min(x)) / 4.0
    yticks = (np.max(y) - np.min(y)) / 4.0	

    plt.scatter(x, y, c=u_inf, cmap="turbo")  # , vmin=umin, vmax=umax)
    plt.colorbar(
        ticks=np.linspace(
            np.min(u_inf) + 1e-6, np.max(u_inf), 5
        )
    )
    #plt.xticks(np.arange(np.min(x), np.max(x) + 1e-6, (np.max(x)-np.min(x))/5), xticks)
    plt.xticks(np.arange(np.min(x), np.max(x) + 1e-6, xticks))
    plt.yticks(np.arange(np.min(y), np.max(y) + 1e-6, yticks))
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title("inference")

    plt.savefig("./results/comparison_" + title + "_" + str(epoch) + ".pdf", dpi=300)
    plt.clf()
    plt.close()
    with open('./results/data_'+ title + "_" + str(epoch) + ".txt"	, 'wb') as f:
        pkl.dump((x,y,u_inf), f)	
        #pkl.dump(data, f)
        #	pkl.dump(data, f)
        #print(str(x), file=f)
        #print(str(y), file=f)
        #print(str(u_inf), file=f)
       
def plot_loss_curve(epoch, logs, labels):
    epoch_log = logs[0]
    plt.figure(figsize=(4, 4))
    for log, label in zip(logs[1:], labels[1:]):
        plt.plot(epoch_log, log, ls="-", alpha=0.7, label=label)  # , c="k")
    # plt.plot(epoch_log, loss_pde_log, ls="--", alpha=.3, label="loss_pde", c="tab:blue")
    # plt.plot(epoch_log, loss_ic_log,  ls="--", alpha=.3, label="loss_ic",  c="tab:orange")
    # plt.plot(epoch_log, loss_bc_log,  ls="--", alpha=.3, label="loss_bc",  c="tab:green")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xscale("linear")
    plt.yscale("log")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig("./results/loss_curve_" + str(epoch) + ".pdf", dpi=300)
    plt.clf()
    plt.close()
