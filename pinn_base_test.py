import os
import numpy as np
import tensorflow as tf

import tensorflow.keras as keras

from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import regularizers, backend


import tfm.nlp.layers.BlockDiagFeedforward as BlockDense

from keras.initializers import VarianceScaling
from keras.layers import Dense
from keras.src import ops, dtype_policies
from utils import eval_dict, replace_words
from lbfgs import lbfgs_minimize, set_LBFGS_options

class PINN(tf.keras.Sequential):
    def __init__(
        self,
        # f_in,
        # f_out,
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

        self.act_func = self.init_act_func(self.act)

        self.dynamic_normalisation = True
        # self.gammas = None
        # seed
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # build a network
        self.add(keras.layers.InputLayer((self.f_in,)))
        for _ in range(self.depth):
            #self.add(ParallelDense(self.f_hid*self.f_out, activation=self.act_func, 
            #                        kernel_initializer=GlorotUniformBlocked(self.f_out, 42)),)#keras.layers.Dense
            self.add(BlockDense(self.f_hid, activation=self.act_func, dropout=0, num_blocks=2))
        self.add(keras.layers.Dense(self.f_out))
        
        '''
        self.denses = []
        # hidden layers
        for i in range(1, len(layer_sizes) - 1):
            prev_layer_size = layer_sizes[i - 1]
            curr_layer_size = layer_sizes[i]
            # Non-Shared layers
            if isinstance(curr_layer_size, (list, tuple)):
                if len(curr_layer_size) != n_output:
                    raise ValueError(
                        "number of sub-layers should equal number of network outputs"
                    )
                # e.g. [8, 8, 8] -> [16, 16, 16] or 64 -> [8, 8, 8]
                self.denses.append(
                    [
                        tf.keras.layers.Dense(
                            units,
                            activation=activation,
                            kernel_initializer=initializer,
                            kernel_regularizer=self.regularizer,
                        )
                        for units in curr_layer_size
                    ]
                )
            # Shared layers
            else:  # e.g. 64 -> 64
                if not isinstance(prev_layer_size, int):
                    raise ValueError(
                        "cannot rejoin parallel subnetworks after splitting"
                    )
                self.denses.append(
                    tf.keras.layers.Dense(
                        curr_layer_size,
                        activation=activation,
                        kernel_initializer=initializer,
                        kernel_regularizer=self.regularizer,
                    )
                )

        # output layers
        if isinstance(layer_sizes[-2], (list, tuple)):  # e.g. [3, 3, 3] -> 3
            self.denses.append(
                [
                    tf.keras.layers.Dense(
                        1,
                        kernel_initializer=initializer,
                        kernel_regularizer=self.regularizer,
                    )
                    for _ in range(n_output)
                ]
            )
        else:
            self.denses.append(
                tf.keras.layers.Dense(
                    n_output,
                    kernel_initializer=initializer,
                    kernel_regularizer=self.regularizer,
                )
            )
        
        '''
        # optimizer (overwrite the learning rate if necessary)
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr, decay_steps=1000, decay_rate=0.9
        )
        
        set_LBFGS_options()
        self.optimizer = lbfgs_minimize
        
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

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
            # try:
            g = eval(eq_string, locals() | self.custom_vars | {"tf": tf})
            # except:
            #    raise ValueError(eq_string)
            g = tf.transpose(tf.convert_to_tensor(g, dtype=tf.float32))
        else:
            u_ = self(vars, training=True)
            self.var_names
            try:
                g = eval(eq_string, locals() | self.custom_vars | {"tf": tf})
            except:  # SyntaxError:
                raise SyntaxError("The line has an error\n" + eq_string)
            g = tf.transpose(tf.convert_to_tensor(g, dtype=tf.float32))
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

    def init_dynamical_normalisastion(self, num_of_losses):
        self.gammas = tf.Variable(tf.ones(num_of_losses), tf.float32)

    @tf.function
    def update_gammas(self, grads):
        def full_reduct(v):
            v = tf.math.abs(v)
            # v = v*v
            while True:
                try:
                    v = tf.reduce_sum(v, axis=1)
                except:
                    break
            return v

        grd = tf.cast([full_reduct(v) for v in grads], tf.float32)
        grd = tf.reduce_sum(tf.math.abs(grd), axis=0)
        # new_gammas = tf.math.reduce_min(grd) * tf.math.divide(tf.ones_like(grd), grd)
        # self.gammas.assign(tf.math.abs(new_gammas)/2)

    @tf.function
    def train(self, conditions, conds_string):
        with tf.GradientTape(persistent=False, watch_accessed_variables=True) as tp:
            #tp.watch(self.trainable_weights)
            losses = tf.cast(eval(conds_string), tf.float32)
            losses_normed = self.normalize_losses(losses)
            grads = tp.jacobian(losses_normed, self.trainable_weights)
        del tp
        #self.update_gammas(grads)
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
        self.optimizer(self.trainable_weights, loss)
        

def block_diagonal(matrices, dtype=tf.float32):
  r"""Constructs block-diagonal matrices from a list of batched 2D tensors.

  Args:
    matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
      matrices with the same batch dimension).
    dtype: Data type to use. The Tensors in `matrices` must match this dtype.
  Returns:
    A matrix with the input matrices stacked along its main diagonal, having
    shape [..., \sum_i N_i, \sum_i M_i].

  """
  matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
  blocked_rows = tf.Dimension(0)
  blocked_cols = tf.Dimension(0)
  batch_shape = tf.TensorShape(None)
  for matrix in matrices:
    full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
    batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
    blocked_rows += full_matrix_shape[-2]
    blocked_cols += full_matrix_shape[-1]
  ret_columns_list = []
  for matrix in matrices:
    matrix_shape = tf.shape(matrix)
    ret_columns_list.append(matrix_shape[-1])
  ret_columns = tf.add_n(ret_columns_list)
  row_blocks = []
  current_column = 0
  for matrix in matrices:
    matrix_shape = tf.shape(matrix)
    row_before_length = current_column
    current_column += matrix_shape[-1]
    row_after_length = ret_columns - current_column
    row_blocks.append(tf.pad(
        tensor=matrix,
        paddings=tf.concat(
            [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
             [(row_before_length, row_after_length)]],
            axis=0)))
  blocked = tf.concat(row_blocks, -2)
  blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
  return blocked


def compute_fans(shape):
    """Computes the number of input and output units for a weight shape.

    Args:
        shape: Integer shape tuple.

    Returns:
        A tuple of integer scalars: `(fan_in, fan_out)`.
    """
    shape = tuple(shape)
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)
    

from keras.src.backend import random
class GlorotUniformBlocked(VarianceScaling):
    def __init__(self, n_blocks=1, seed=None):
        super().__init__(
            scale=1.0, mode="fan_avg", distribution="uniform", seed=seed
        )
        self.n_blocks = n_blocks
    
    def __call__(self, shape, dtype=None):
        scale = self.scale
        #total_shape = (np.prod(shape[0]),np.prod(shape[1])) 
        fan_in, fan_out = compute_fans(shape)
        scale /= max(1.0, fan_in)
        limit = tf.math.sqrt(3.0 * scale)
        base_init = random.uniform(
            shape, minval=-limit, maxval=limit, dtype=dtype, seed=self.seed
        )
        #small_shape = np.array((np.array(shape)/self.n_blocks), dtype=int)
        #blocked = block_diagonal([ops.ones(shape) for i in range(self.n_blocks)])
        
        linop_blocks = (ops.ones(shape) for i in range(self.n_blocks))
        blocked = tf.linalg.LinearOperatorBlockDiag(linop_blocks, is_non_singular=False)

        print(blocked.to_dense())

        return blocked * base_init

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }

class ParallelDense(Dense):
    def build(self, input_shape):
        kernel_shape = (input_shape[-1], self.units)
        # We use `self._dtype_policy` to check to avoid issues in torch dynamo
        #is_quantized = isinstance(
        #    self._dtype_policy, dtype_policies.QuantizedDTypePolicy
        #)
        #if is_quantized:
        #    self.quantized_build(
        #        input_shape, mode=self.dtype_policy.quantization_mode
        #    )
        #if not is_quantized or self.dtype_policy.quantization_mode != "int8":
            # If the layer is quantized to int8, `self._kernel` will be added
            # in `self._int8_build`. Therefore, we skip it here.
        print(self.units)   
        self._kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_shape[-1]})
        self.built = True
        if self.lora_rank:
            self.enable_lora(self.lora_rank)
    
    def add_weight(
        self,
        shape=None,
        initializer=None,
        dtype=None,
        trainable=True,
        autocast=True,
        regularizer=None,
        constraint=None,
        aggregation="none",
        overwrite_with_gradient=False,
        name=None,
    ):
        """Add a weight variable to the layer.

        Args:
            shape: Shape tuple for the variable. Must be fully-defined
                (no `None` entries). Defaults to `()` (scalar) if unspecified.
            initializer: Initializer object to use to populate the initial
                variable value, or string name of a built-in initializer
                (e.g. `"random_normal"`). If unspecified, defaults to
                `"glorot_uniform"` for floating-point variables and to `"zeros"`
                for all other types (e.g. int, bool).
            dtype: Dtype of the variable to create, e.g. `"float32"`. If
                unspecified, defaults to the layer's variable dtype
                (which itself defaults to `"float32"` if unspecified).
            trainable: Boolean, whether the variable should be trainable via
                backprop or whether its updates are managed manually. Defaults
                to `True`.
            autocast: Boolean, whether to autocast layers variables when
                accessing them. Defaults to `True`.
            regularizer: Regularizer object to call to apply penalty on the
                weight. These penalties are summed into the loss function
                during optimization. Defaults to `None`.
            constraint: Contrainst object to call on the variable after any
                optimizer update, or string name of a built-in constraint.
                Defaults to `None`.
            aggregation: Optional string, one of `None`, `"none"`, `"mean"`,
                `"sum"` or `"only_first_replica"`. Annotates the variable with
                the type of multi-replica aggregation to be used for this
                variable when writing custom data parallel training loops.
                Defaults to `"none"`.
            overwrite_with_gradient: Boolean, whether to overwrite the variable
                with the computed gradient. This is useful for float8 training.
                Defaults to `False`.
            name: String name of the variable. Useful for debugging purposes.
        """
        self._check_super_called()
        if shape is None:
            shape = ()
        if dtype is not None:
            dtype = backend.standardize_dtype(dtype)
        else:
            dtype = self.variable_dtype
        if initializer is None:
            if "float" in dtype:
                initializer = "glorot_uniform"
            else:
                initializer = "zeros"
        initializer = initializers.get(initializer)
        with backend.name_scope(self.name, caller=self):
            variable = backend.Variable(
                initializer=initializer,
                shape=shape,
                dtype=dtype,
                trainable=trainable,
                autocast=autocast,
                aggregation=aggregation,
                name=name,
            )
        # Will be added to layer.losses
        variable.regularizer = regularizers.get(regularizer)
        variable.constraint = constraints.get(constraint)
        variable.overwrite_with_gradient = overwrite_with_gradient
        self._track_variable(variable)
        return variable
