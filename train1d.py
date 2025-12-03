"""
********************************************************************************
training
********************************************************************************
"""
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time
import yaml
import numpy as np 
import tensorflow as tf
import pickle
from tqdm import tqdm

from config_gpu import config_gpu
from pinn_base import PINN
from si_pinn import SI_PINN
from utils import (
    make_logger,
    write_logger,
    eval_dict,
    gen_condition,
    plot_loss_curve,
    plot_comparison1d,
    to_gif
)


def train1d(filename, model_class, output_dir=""):
    # read settings
    with open(filename, mode="r") as file:
        settings = yaml.safe_load(file)

    # run hyperparameters args
    logger_path = make_logger("seed: in model", output_dir=output_dir)
    args = eval_dict(settings["ARGS"])

    # ======model=======

    model_args = eval_dict(settings["MODEL"], {"tf": tf, "": np})

    for key in model_args.keys():
        if isinstance(model_args[key], list):
            model_args[key] = tf.constant(model_args[key], tf.float32)

    var_names = settings["IN_VAR_NAMES"]
    func_names = settings["OUT_VAR_NAMES"]
    
    model = model_class(var_names=var_names, func_names=func_names, **(model_args))
    model.init_custom_vars(
        dict_consts=settings["CUSTOM_CONSTS"],
        dict_funcs=settings["CUSTOM_FUNCS"],
        var_names=var_names,
    )

    # print("INIT DONE")

    in_lb = model_args["in_lb"]
    tmin = in_lb[0]
    in_ub = model_args["in_ub"]
    tmax = in_ub[0]
    # ======conditions=======

    conds = eval_dict(settings["CONDS"], locals() | {"tf": tf} | model.custom_vars, 1)
    conditions = []
    for key in list(conds.keys()):
        cond_ = gen_condition(
            conds[key], model_args, func_names=func_names, var_names=var_names, **model.custom_vars
        )
        conditions.append(cond_)
    cond_string = [
        "self.loss_(*conditions[" + str(i) + "])," for i in range(len(conditions))
    ]
    cond_string = compile("(" + "".join(cond_string) + ")", '<string>', 'eval')
    
    model.init_dynamical_normalisation(len(conditions))

    # # ======outputs=======

    ns = eval_dict(settings["NS"])
    _x = [0] * len(var_names)
    for i in range(len(var_names)):
        _x[i] = tf.linspace(in_lb[i], in_ub[i], ns["nx"][i])
    _x = (tf.meshgrid(*_x))

    
    x = [0]*len(_x)
    for i in range(len(var_names)):
        x[i] = tf.reshape(_x[i],(-1,1))
    x_ref = tf.transpose(tf.cast(x, dtype=tf.float32))[0]
    u_ref = tf.cast(np.zeros(ns['nx']).reshape(-1, 1), dtype=tf.float32)
    exact = tf.cast(model.custom_vars["exact"](x_ref), dtype=tf.float32)
    
    # log
    losses_logs = np.empty((len(conds.keys()), 1))

    # training
    wait = 0
    loss_best = tf.constant(1e20)
    loss_save = tf.constant(1e20)
    t0 = time.perf_counter()

    cond_string_here = [
       "model.loss_(*conditions[" + str(i) + "])," for i in range(len(conditions))
    ]
    cond_string_here = "(" + "".join(cond_string_here) + ")"

    # print("START TRAINING")
    for epoch in tqdm(range(1, int(args["epochs"]) + 1)):
        # gradient descent
        loss_glb, losses = model.train(conditions, cond_string)
        
        losses_logs = np.append(losses_logs, np.expand_dims(losses, axis=0).T, axis=1)
        
        t1 = time.perf_counter()
        elps = t1 - t0
        # print(elps)
        # print(loss_glb)
        losses = dict(zip(conds.keys(), losses))
        logger_data = [key + f": {losses[key]:.3e}, " for key in losses.keys()]
        logger_data = f"epoch: {epoch:d}, loss_total: {loss_glb:.3e}, " + ", ".join(
            logger_data
        )
        write_logger(logger_path, logger_data)
        # print(logger_data)
            
        # early stopping
        if loss_glb < loss_best * 1.5:
            loss_best = loss_glb
            wait = 0
        else:
            if wait >= args["patience"]:
                print(">>>>> early stopping")
                break
            wait += 1

        # monitor
        if epoch % 1000 == 0:
            file_extension = "jpg"
            u_ = model(x_ref)
            u_n = u_.numpy().transpose()
            # print("Estimation error ", np.max(np.abs(exact - u_[:, 0])))
            plot_commons = {
                "epoch": epoch,
                "x": x_ref[:, 0],
                "y": None,#x_ref[:, 1],
                "xlabel": var_names[0],
                "ylabel": None,#var_names[1],
            }
            for func, title in zip(u_n, func_names):
                plot_comparison1d(u_inf=func, 
                                  title=title, 
                                  file_extension=file_extension, 
                                  output_dir=output_dir,
                                  **plot_commons)
            
            plot_loss_curve(epoch, 
                            losses_logs[:, 1:], 
                            labels=list(conds.keys()), 
                            file_extension=file_extension,
                            output_dir=output_dir,)
    for title in func_names:
        to_gif(["./results" + output_dir + "/comparison_" + title + "_" + str(ep) + "." + file_extension \
                     for ep in range(1000, (epoch // 1000 + 1) * 1000, 1000)],
                     output_gif="./results" + output_dir + "/comparison_" + title + ".gif",
                     duration=150,
                     dpi=300,
                     sort_files=False)
            
if __name__ == "__main__":
    config_gpu(flag=-1, verbose=True)
    train1d(filename="./settings/simple-sir.yaml", model_class=SI_PINN, output_dir="/simple-sir/si_pinn")
