import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

from model import UNet
from loss import FDLoss, TimeDependentLoss

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)


# Elliptic problems
# Set grid size
N = 128
h = 1./(N - 1)

# Define grid
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
[X, Y] = np.meshgrid(x, y)

# Get problem data
# Bubble function
u = X*(X - 1)*Y*(Y - 1)
f = 2*(X - 1)*X + 2*(Y - 1)*Y

# Boundary data
g = [u[:, 0], u[:, -1], u[0, :], u[-1, :]]

# Define loss function
loss_fn = FDLoss(N, f, g)

# Run training loop for elliptic problem
n_steps = 2000
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps=n_steps
)
optimizer = tf.keras.optimizers.Adam(lr_schedule)
optimizer.global_clipnorm = 0.001

model = UNet(1, 32, 3, True)

inp = f.reshape(1, N, N, 1)

prog_bar = Progbar(n_steps, stateful_metrics=['loss'])

@tf.function
def train_step(input):
    with tf.GradientTape() as tape:
        p = model(inp)
        loss = loss_fn(p, p)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, p

best = np.Inf
best_p = np.zeros((1, N, N, 1))
print("Solving elliptic problem...")
for step in range(n_steps):
    loss, p = train_step(input)
    prog_bar.add(1, values=[('loss', loss)])

    if loss < best:
        best = loss
        best_p = p

pred = best_p.numpy().reshape(N, N)

print("L2 error: {}".format(np.format_float_scientific(h*np.sqrt(np.sum(np.square(u - pred))), 4)))
print("L-inf error: {}".format(np.format_float_scientific(np.max(np.abs(u - pred)), 4)))
print(" ")
print("L2 error from paper: 4.0090e-05")
print("L-inf error from paper: 3.0874e-04")

# Parabolic/time dependent problem
# Define time step
time_step = 0.1

# Trigonometric functions (we used n = 1 and n = 4)
n = 1
u = lambda t: tf.constant((np.cos(t) * np.sin(n * np.pi * X) * np.sin(n * np.pi * Y)).reshape(1, N, N, 1), dtype=tf.float32)
f = lambda t: tf.constant((-np.sin(n*np.pi*X)*np.sin(n*np.pi*Y)*(-2*np.cos(t)*n**2*np.pi**2 + np.sin(t))).reshape(1, N, N, 1), dtype=tf.float32)

# Initial condition
u0 = u(0)

# Run training loop for parabolic/time dependent problem
loss_fn = TimeDependentLoss(N, time_step, f)

u_previous = u0
_n_steps = 250
model = UNet(1, 32, 3, True)

initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=initial_learning_rate,
    first_decay_steps=_n_steps,
    t_mul=1.0,
    m_mul=1.0,
    alpha=0.0
)
optimizer = tf.keras.optimizers.Adam(lr_schedule)
optimizer.global_clipnorm = 0.001

@tf.function
def train_step(u_previous, f, t):
    with tf.GradientTape() as tape:
        p = model(u_previous)
        loss = loss_fn((p, u_previous), t)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, p

sol = list()
sol.append(u_previous.numpy().reshape(N, N))

print("Solving parabolic/time dependent problem...")
for t in range(10):
    best = np.Inf
    if t == 0:
        n_steps = 4*_n_steps
    else:
        n_steps = _n_steps

    prog_bar = Progbar(n_steps, stateful_metrics=["time_step", "loss"])
    for step in range(n_steps):
        loss, u_candidate = train_step(u_previous, f, time_step*(1 + t))
        prog_bar.add(1, values=[("time_step", int(t + 1)), ("loss", loss)])

        if loss < best:
            best = loss
            u_next = u_candidate

    u_previous = u_next
    sol.append(u_next.numpy().reshape(N, N))


step = 5
print("L2 error at t = 0.5: {}".format(np.format_float_scientific(h*np.sqrt(np.sum(np.square(u(time_step * step).numpy().reshape(N, N) - sol[step]))), 4)))
print("L-inf error at t = 0.5: {}".format(np.format_float_scientific(np.max(np.abs(u(time_step * step).numpy().reshape(N, N) - sol[step])), 4)))
print(" ")
print("L2 error at t = 0.5 from paper: 1.1562e-03")
print("L-inf error at t = 0.5 from paper: 2.4109e-03")