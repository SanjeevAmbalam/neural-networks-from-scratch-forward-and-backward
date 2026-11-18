"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
import numpy as np
def numerical_gradient(f, x, eps=1e-5):
    # TODO: Estimate the gradient of scalar f w.r.t. array x via central finite differences
    grad = np.zeros(x.shape)
    for i in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus))/(2*eps)
    return grad

# Step 2 - gradient_check
import numpy as np
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # TODO: Return max relative error between analytic and numeric gradients.
    anal = np.asarray(analytic_grad)
    num = np.asarray(numeric_grad)
    diff = np.abs(num-anal)
    max_vals = np.maximum(np.abs(anal), np.abs(num))
    err = diff/np.maximum(max_vals, tol)

    return float(np.max(err))

# Step 3 - make_dense
import numpy as np
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    # TODO: your approach here
    W, b = weight_init_fn(in_dim, out_dim)
    params = {'W': W, 'b': b}
    
    def forward(x):
      y = x @ params['W'] + params['b']
      cache = x
      return y, cache

    def backward(dout, cache):
      x = cache
      dx = dout @ params['W'].T
      dW = x.T @ dout
      db = np.sum(dout, axis=0)

      grads = {'W': dW, 'b': db}
      return dx, grads

    return{
    'params': params,
    'forward': forward,
    'backward': backward
    }

# Step 4 - make_activation
import numpy as np
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    # TODO: your approach here
    def forward(x):
      y = np.maximum(x, 0)
      cache = x
      return y, cache

    def backward(dout, cache):
      dx = (cache>0) * dout
      return dx, {}

    return {
      'params': {},
      'forward': forward,
      'backward': backward
    }

# Step 5 - initialize_weights
import numpy as np
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    # TODO: your approach here
    W = np.random.randn(in_dim, out_dim) * np.sqrt(2 / in_dim)
    b = np.zeros(out_dim,)

    return W, b

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits)."""
    def loss_fn(logits, labels):
        # apply softmax
        N, C = logits.shape

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        prob = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        one_hot = np.eye(C)[labels]

        loss = np.mean(np.log(np.sum(exp_z, axis=1))-shifted[np.arange(N), labels])
        d_logits = (prob-one_hot)/N

        return float(loss), d_logits

    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    # TODO: your approach here
    def forward(x):
      caches = []
      for layer in layers:
        x, cache = layer['forward'](x)
        caches.append(cache)
      return x, caches

    def backward(dout, caches):
      grads_list = [{} for layer in layers]
      for i in reversed(range(len(layers))):
        dout, grads = layers[i]['backward'](dout, caches[i])
        grads_list[i] = grads
      return dout, grads_list
    
    params = [layer['params'] for layer in layers]
    return {'forward': forward, 'backward': backward, 'params': params}

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

