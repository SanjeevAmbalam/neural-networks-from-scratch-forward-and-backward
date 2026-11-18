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

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    # TODO: your approach here
    logits, caches = model['forward'](x)
    loss, d_logits = loss_fn(logits, y)

    dx, param_grads = model['backward'](d_logits, caches)
    return loss, param_grads

# Step 9 - make_optimizer
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    # TODO: your approach here
    def update(p, g):
        if isinstance(p, dict):
            for key in p:
                update(p[key], g[key])

        elif isinstance(p, (list, tuple)):
            for pi, gi in zip(p, g):
                update(pi, gi)

        else:
            p -= lr * g

    def step(grads):
        update(params, grads)

    return {'step': step}

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    # TODO: your approach here
    params = model['params']
    loss, param_grads = forward_backward(model, loss_fn, x_batch, y_batch)
    optimizer['step'](param_grads)

    return loss

# Step 11 - train
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    # TODO: your approach here
    history = []
    rng = np.random.RandomState(seed)
    N = len(x)

    for t in range(epochs):
        indices = rng.permutation(N)
        losses = []
        for i in range(0, N, batch_size):
            batch_indices = indices[i:i + batch_size]

            x_batch = x[batch_indices]
            y_batch = y[batch_indices]

            loss = train_step(model, loss_fn, optimizer, x_batch, y_batch)
            losses.append(loss)

        history.append(float(np.mean(losses)))

    return history

# Step 12 - design_network
def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task.

    Inputs:
      input_dim: int, feature dimension
      num_classes: int, number of classes
      seed: int, RNG seed for reproducibility

    Returns:
      model: trained sequential model (forward/backward/params)
      metrics: dict with
        'accuracy': float >= 0.90 on an evaluation set,
        'x': np.ndarray (N, input_dim) eval features (N >= 50),
        'y': np.ndarray (N,) integer eval labels.
      The eval set (x, y) must not be linearly separable to high accuracy
      (< 0.82 for a linear classifier), and the model's true accuracy on
      it must match metrics['accuracy'] and be >= 0.90.
    """
    # TODO: your approach here
    if input_dim < 1 or num_classes < 2:
        raise ValueError("Invalid dimensions")

    rng = np.random.RandomState(seed)

    def make_data(N):
        labels = np.arange(N) % num_classes
        rng.shuffle(labels)

        signs = rng.choice([-1.0, 1.0], size=N)
        radius = 1.0 + 1.5 * labels

        x = rng.normal(0, 0.12, size=(N, input_dim))
        x[:, 0] += signs * radius

        return x, labels.astype(int)

    x_train, y_train = make_data(768)
    x_eval, y_eval = make_data(240)

    def weight_init(in_dim, out_dim):
        W = rng.randn(in_dim, out_dim) * np.sqrt(2 / in_dim)
        b = np.zeros(out_dim)
        return W, b

    layers = [
        make_dense(input_dim, 64, weight_init),
        make_activation('relu'),
        make_dense(64, 64, weight_init),
        make_activation('relu'),
        make_dense(64, num_classes, weight_init)
    ]

    model = make_sequential(layers)
    loss_fn = make_loss('cross_entropy')
    optimizer = make_optimizer(model['params'], lr=0.02)

    history = train(model, loss_fn, optimizer,x_train, y_train, epochs=100, batch_size=32, seed=seed)

    logits, caches = model['forward'](x_eval)
    predictions = np.argmax(logits, axis=1)
    accuracy = float(np.mean(predictions == y_eval))

    metrics = {'accuracy': accuracy, 'x': x_eval, 'y': y_eval}
    return model, metrics

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

