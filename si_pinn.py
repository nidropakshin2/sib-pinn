import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
import yaml
import tensorflow.keras as keras
import traceback


from pinn_base import PINN
from utils import (
    eval_dict, 
    replace_words,
    make_logger,
    eval_dict,
)



class WaveAct(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.w1 = self.add_weight(
            name="w1",
            shape=(),
            initializer="random_normal",
            trainable=True
        )
        self.w2 = self.add_weight(
            name="w2", 
            shape=(),
            initializer="random_normal",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return self.w1 * tf.math.sin(inputs) + self.w2 * tf.math.cos(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


def TestWaveAct(show=False):
    try:
        act_func = WaveAct()
        x = tf.Variable([1.3, 0.7, 2.4, 0.5, -1.1])
        with tf.GradientTape(persistent=True) as tape:    
            y = act_func(x)
            grad = tape.gradient(y, x)
            if show:
                tf.print(y, grad, sep="\n")
        del tape
        return True
    except:
        return False


class FourierPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, f_in, f_hid, f_out, **kwargs):
        super().__init__(**kwargs)

        self.f_in = f_in
        self.f_out = f_out
        self.f_hid = f_hid
        self.B = tf.random.uniform((f_in, self.f_hid))

    def build(self, f_in):
        self.fourier_emb = self.add_weight(
            name="Fourier",
            shape=(2 * self.f_hid, self.f_out),
            initializer="random_normal",
            trainable=True
        )

        self.pos_emb = self.add_weight(
            name="Positional",
            shape=(self.f_in, self.f_out),
            initializer="random_normal",
            trainable=True
        )
        
        super().build(self.f_in)

    def call(self, x):
        
        z = 2 * np.pi * tf.matmul(x, self.B)
        f = tf.concat([tf.sin(z), tf.cos(z)], axis=-1)
        
        emb_f = tf.matmul(f, self.fourier_emb)
        emb_p = tf.matmul(x, self.pos_emb)
        
        return emb_f + emb_p
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.f_out)


def TestFourierPositionalEmbedding(show=False):
    try:
        x = tf.transpose(tf.random.uniform((2, 5), dtype=tf.double))
        t = tf.transpose([tf.linspace(start=0, stop=1, num=5)])
        spacetime = tf.concat([t, x], axis=-1)
        FourierLayer = FourierPositionalEmbedding(3, 4, 4)
    
        with tf.GradientTape(persistent=True) as tape:    
            y = FourierLayer(spacetime)
            grad = tape.gradient(y, x)
            if show:
                tf.print(y, grad, sep="\n")
        del tape
        return True
    except Exception as e:
        print(f"\n{type(e).__name__}: {e}\n")
        traceback.print_exc() 
        return False


class WaveletLayer(tf.keras.layers.Layer):
    def __init__(self, f_in, f_out, **kwargs):
        super().__init__(**kwargs)
        self.f_in = f_in
        self.f_out = f_out
        
    def build(self, f_in):
        self.lin = self.add_weight(
            name="Linear",
            shape=(self.f_in, self.f_out),
            initializer="random_uniform",
            trainable=True
        )
        self.act = WaveAct()
        super().build(f_in)
    
    def call(self, x):
        out = self.act(tf.matmul(x, self.lin))
        return out


def TestWaveletLayer(show=False):
    try:
        x = tf.transpose(tf.random.uniform((2, 5), dtype=tf.double))
        t = tf.transpose([tf.linspace(start=0, stop=1, num=5)])
        spacetime = tf.concat([t, x], axis=-1)
        Wavelet = WaveletLayer(f_in=3, f_out=2)
    
        with tf.GradientTape(persistent=True) as tape:    
            y = Wavelet(spacetime)
            grad = tape.gradient(y, x)
            if show:
                tf.print(y, grad, sep="\n")
        del tape
        return True
    except Exception as e:
        print(f"\n{type(e).__name__}: {e}\n")
        traceback.print_exc() 
        return False

class SI_PINN(tf.keras.Sequential):
    def __init__(
        self,
        f_hid,
        depth,
        in_lb,
        in_ub,
        var_names,
        func_names,
        w_init="Glorot",
        b_init="zeros",
        act="tanh",
        lr=1e-3,
        dyn_norm="max_avg",
        beta=0.01,
        seed=42,
    ):
        super().__init__()
        self.var_names = var_names
        self.f_in = int(len(var_names))  # f_in)
        self.f_out = int(len(func_names))  # f_out)
        self.f_hid = int(f_hid)
        self.depth = int(depth)
        self.lb = in_lb  # lower bound of input
        self.ub = in_ub  # upper bound of input
        self.mean = (in_lb + in_ub) / 2
        self.w_init = w_init  # weight initialization
        self.b_init = b_init  # bias initialization
        self.act = act  # activation
        self.lr = lr  # learning rate
        self.seed = int(seed)
        self.f_scl = "minmax"  # "linear" / "minmax" / "mean"
        self.d_type = tf.float32
        self.model_name = "si_pinn"

        self.act_func = self.init_act_func(self.act)

        # Note that it assumes that first element of var_names belongs to pde
        self.dynamic_normalisation = dyn_norm
        if 0 <= beta and beta <= 1: 
            self.beta = beta
        else:
            raise ValueError("parameter beta must be between 0 and 1")

        # seed
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # build a network
        self.add(FourierPositionalEmbedding(f_in=self.f_in, 
                                            f_hid=self.f_hid, 
                                            f_out=self.f_hid, 
                                            input_shape=(self.f_in,)))
        # self.add(WaveletLayer(f_in=self.f_hid, f_out=self.f_hid))
        for _ in range(self.depth):
            self.add(keras.layers.Dense(self.f_hid, activation=self.act_func))
        self.add(keras.layers.Dense(self.f_out))

        # optimizer (overwrite the learning rate if necessary)
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr, decay_steps=3000, decay_rate=0.7
        )
        
        #set_LBFGS_options()
        #self.optimizer = 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # system params
        self.custom_vars = {}

    def _inner_lambda(self, _dict_func, var_names: list, other_dicts={}):  # _variables
        inner_vars_dict = {}
        for name in var_names:
            inner_vars_dict[name] = eval(name)
        return eval(_dict_func, other_dicts | inner_vars_dict)

    def init_custom_vars(
        self, dict_consts: dict, dict_funcs: dict = {}, var_names: list = []
    ):
        def make_lambda(string):
            string = compile(string, "<string>", "eval",optimize=1)
            return lambda _variables: eval(
                string, self.custom_vars | {"tf": tf, "_variables": _variables}
            )

        self.custom_vars = eval_dict(dict_consts, {"tf": tf})
        for key in self.custom_vars.keys():
            self.custom_vars[key] = tf.constant(self.custom_vars[key])

        replecement_dict = {}
        for i in range(len(var_names)):
            replecement_dict[var_names[i]] = "_variables[:," + str(i) + "]"
        for key in dict_funcs.keys():
            self.custom_vars.update({
                key: make_lambda(replace_words(dict_funcs[key], replecement_dict))
            })

    def init_act_func(self, act):
        if act == "tanh":
            return lambda u: tf.math.tanh(u)
        elif act == "softplus":
            return lambda u: tf.math.softplus(u)
        elif act == "silu" or act == "swish":
            return lambda u: tf.multiply(u, tf.math.sigmoid(u))
        elif act == "gelu":
            return lambda u: tf.multiply(u, tf.math.sigmoid(1.702 * u))
        elif act == "mish":
            return lambda u: tf.multiply(u, tf.math.tanh(tf.math.softplus(u)))
        else:
            raise NotImplementedError(">>>>> forward_pass (act)")

    @tf.function#(jit_compile=True)
    def compute_pde(self, vars, eq_string, compute_grads=False):
        if compute_grads:
            with tf.GradientTape(
                persistent=False, watch_accessed_variables=True
            ) as tp1:
                #tp1.watch(vars)
                with tf.GradientTape(
                    persistent=True, watch_accessed_variables=True
                ) as tp2:
                    #tp2.watch(vars)
                    u_ = self(vars, training=True)
                u_x = tp2.batch_jacobian(u_, vars)
                del tp2
            u_xx = tp1.batch_jacobian(u_x, vars)
            del tp1
            g = eval(eq_string, locals() | self.custom_vars | {"tf": tf})
        else:
            u_ = self(vars, training=True)
            g = eval(eq_string, locals() | self.custom_vars | {"tf": tf})    
        g = tf.convert_to_tensor(g, dtype=tf.float32)
        return u_, g

    @tf.function
    def std_error(self, vals, exact_vals):
        return tf.reduce_mean(tf.square(vals - exact_vals))

    @tf.function
    def loss_(self, x, exact_vals, eq_string, compute_grads):
        _, g_ = self.compute_pde(x, eq_string, compute_grads)
        loss = self.std_error(g_, exact_vals)
        return loss

    # def infer(self, x):
    #    u_, g_ = self.compute_pde(x, compute_grads=False)
    #    return u_, g_

    @tf.function
    def normalize(self, input_vector):
        vector0 = tf.identity(input_vector)
        vector = tf.identity(vector0)
        return input_vector * tf.math.reduce_max(vector) / vector

    def normalize_losses(self, vec):
        return vec * self.gammas

    def init_dynamical_normalisation(self, num_of_losses):
        self.gammas = tf.Variable(tf.ones(num_of_losses), tf.float32)

    @tf.autograph.experimental.do_not_convert
    def update_gammas(self, grads):
        # TODO: реализовать проверку совпадения первых размерностей всех градиентов
        # dims = set([tf.shape(v)[0] for v in grads])
        # if len(dims) > 1:
        #     raise ValueError(f"All tensors must have common first dimension, got: {dims}")
        
        if self.dynamic_normalisation:
            # Приведем градиенты в векторный вид, 
            # т.е. теперь все производные выстроены в одну строчку длины,
            # равной суммарному кол-ву обучаемых параметров модели
        
            grd = tf.concat([tf.reshape(v, [tf.shape(v)[0], -1]) for v in grads], axis=1)
            
            # We assume here that the first grad is from PDE loss
            match self.dynamic_normalisation:
                case "max_avg":
                    grd_mean_abs = tf.reduce_mean(tf.abs(grd), axis=1)
                    gammas_cup = tf.reduce_max(tf.abs(grd[0])) * tf.divide(tf.ones_like(grd_mean_abs), self.gammas * grd_mean_abs)

                case "inv_dir":
                    grd = tf.math.reduce_std(grd, axis=1)
                    gammas_cup = tf.reduce_max(grd) * tf.divide(tf.ones_like(grd), grd)
        
                case "dyn_norm": 
                    grd = tf.norm(grd, axis=1)
                    gammas_cup = tf.reduce_max(grd) * tf.divide(tf.ones_like(grd), grd)
                
                case None:
                    gammas_cup = self.gammas
                
                case _:
                    raise NotImplementedError(f"update_gammas has no dynamical normalisation option '{self.dynamic_normalisation}'")
            self.gammas.assign(self.beta * gammas_cup + (1 - self.beta) * self.gammas)

    @tf.function
    def train(self, conditions, conds_string):
        with tf.GradientTape(persistent=False, watch_accessed_variables=True) as tp:

            losses = tf.cast(eval(conds_string), tf.float32)
            losses_normed = self.normalize_losses(losses)
            
            grads = tp.jacobian(losses_normed, self.trainable_weights)
        del tp
        self.update_gammas(grads)
        loss_glb = tf.math.reduce_sum(losses_normed)
        grad = [tf.reduce_sum(v, axis=0) for v in grads]
        self.optimizer.apply_gradients(zip(grad, self.trainable_weights))
        return loss_glb, losses
    
    @tf.function
    def eval_loss(self, conditions, conds_string):
        losses = tf.cast(eval(conds_string), tf.float32)
        losses_normed = tf.reduce_sum(self.normalize_losses(losses))
        return losses_normed
        
    #@tf.function
    def train_lbfgs(self, conditions, conds_string):
        loss = lambda: self.eval_loss(conditions, conds_string)
        res = lbfgs_minimize(self.trainable_weights, loss)
        return res



class SI_PINN_experimental(PINN):
    def __init__(  
        self,
        f_hid,
        depth,
        in_lb,
        in_ub,
        var_names,
        func_names,
        w_init="Glorot",
        b_init="zeros",
        act="tanh",
        lr=1e-3,
        dynamic_normalisation=None,
        beta=0.01,
        seed=42,
    ):
        super().__init__(
            f_hid,
            depth,
            in_lb,
            in_ub,
            var_names,
            func_names,
            w_init,
            b_init,
            act,
            lr,
            dynamic_normalisation,
            beta,
            seed
        )

        self.layers = []

        # self.add(keras.layers.InputLayer((self.f_in,)))
        self.add(FourierPositionalEmbedding(self.f_in, f_hid=self.f_hid, f_out=self.f_out))
        for _ in range(self.depth):
            self.add(keras.layers.Dense(self.f_hid, activation=self.act_func))
        self.add(WaveletLayer(f_in=self.f_hid, f_hid=self.f_hid, f_out=self.f_hid, depth=2))
        self.add(keras.layers.Dense(self.f_out))


def TestSI_PINN(show=False):
    try:
        filename = "./settings/sir-controlled (Copy).yaml"

        # read settings
        with open(filename, mode="r") as file:
            settings = yaml.safe_load(file)

        # run hyperparameters args
        logger_path = make_logger("seed: in model")
        args = eval_dict(settings["ARGS"])

        # ======model=======

        model_args = eval_dict(settings["MODEL"], {"tf": tf, "": np})

        for key in model_args.keys():
            if isinstance(model_args[key], list):
                model_args[key] = tf.constant(model_args[key], tf.float32)

        var_names = settings["IN_VAR_NAMES"]
        func_names = settings["OUT_VAR_NAMES"]
        model = SI_PINN(var_names=var_names, func_names=func_names, **(model_args))
        model.init_custom_vars(
            dict_consts=settings["CUSTOM_CONSTS"],
            dict_funcs=settings["CUSTOM_FUNCS"],
            var_names=var_names,
        )

        if show:
            tf.print(model.summary())
        return True
    except Exception as e:
        print(f"\n{type(e).__name__}: {e}\n")
        traceback.print_exc() 
        return False


def testing_wrapper(func_name, test_func, show=False):
    print(f"\nTest {func_name}: ")
    if test_func(show=show):
        print("Successfull\n")
    else:
        print("Failed\n")


if __name__ == "__main__":
    testing_wrapper("WaveAct", TestWaveAct, show=False)
    testing_wrapper("FourierPositionalEmbedding", TestFourierPositionalEmbedding, show=False)
    testing_wrapper("WaveletLayer", TestWaveletLayer, show=False)
    testing_wrapper("SI_PINN", TestSI_PINN, show=True)