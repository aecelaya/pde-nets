import numpy as np
import tensorflow as tf


# Define loss function
class FDLoss(tf.keras.losses.Loss):
    def __init__(self, N, f, g, **kwargs):
        super(FDLoss, self).__init__(**kwargs)
        self.N = N
        self.h = 1./(N - 1)
        self.alpha = np.square(0.5*self.h)

        # Set up right hand side
        self.f = tf.constant(f, dtype=tf.float32)
        self.f = tf.reshape(f, [1, N, N, 1])
        self.f = tf.cast(self.f, tf.float32)

        # Set up finite difference kernel
        k_laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / np.square(self.h)
        k_laplacian = tf.constant(k_laplacian, dtype=tf.float32)
        self.k_laplacian = tf.reshape(k_laplacian, [3, 3, 1, 1])

        # Define boundary terms
        self.g = [tf.cast(g[i], tf.float32) for i in range(4)]

    def call(self, y_true, y_pred):
        # Loss on interior
        rhs = -tf.nn.convolution(y_pred, self.k_laplacian, strides=1)
        interior = tf.reduce_mean(tf.square(rhs + self.f[:, 1:-1, 1:-1, :]))

        # Get boundary values for left, right, bottom, and top
        left_boundary = tf.square(self.g[0] - tf.reshape(y_pred[:, :, 0, :], [self.N]))
        right_boundary = tf.square(self.g[1] - tf.reshape(y_pred[:, :, -1, :], [self.N]))
        bottom_boundary = tf.square(self.g[2] - tf.reshape(y_pred[:, 0, :, :], [self.N]))
        top_boundary = tf.square(self.g[3] - tf.reshape(y_pred[:, -1, :, :], [self.N]))

        # Define boundary loss for left, right, bottom, and top boundaries
        boundary = tf.concat([left_boundary,
                              right_boundary,
                              bottom_boundary,
                              top_boundary], axis = -1)
        boundary = tf.reduce_mean(boundary)

        # Compute final loss
        loss = self.alpha*interior + (1. - self.alpha)*boundary

        return loss
    

# Define loss function
class TimeDependentLoss(tf.keras.losses.Loss):
    def __init__(self, N, step_size, f, **kwargs):
        super(TimeDependentLoss, self).__init__(**kwargs)
        self.N = N
        self.h = 1./(N - 1.)
        self.step_size = step_size
        
        # Tune this parameter
        self.alpha = np.square(self.h) * 4
        
        # Get source term
        self.f = f

        # Set up kernels
        # Laplacian kernel
        k_laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / np.square(self.h)
        k_laplacian = tf.constant(k_laplacian, dtype=tf.float32)
        self.k_laplacian = tf.reshape(k_laplacian, [3, 3, 1, 1])

    def call(self, current_previous, t):

        # Unpack current and previous predictions
        u_current, u_previous = current_previous
        
        # Get value of f at time t
        f_current = self.f(t)[:, 1:-1, 1:-1, :]
        
        # Loss on interior
        u_current_interior = u_current[:, 1:-1, 1:-1, :]
        u_previous_interior = u_previous[:, 1:-1, 1:-1, :]

        # Estimate right hand side (i.e., laplacian(u)) for current step
        rhs = tf.nn.convolution(u_current, self.k_laplacian, strides=1)

        interior = tf.reduce_mean(tf.square(u_current_interior - u_previous_interior - self.step_size*(rhs + f_current)))

        # Loss on boundary
        # Get boundary values for left, right, bottom, and top
        left_boundary = tf.square(tf.reshape(u_current[:, :, 0, :], [self.N]))
        right_boundary = tf.square(tf.reshape(u_current[:, :, -1, :], [self.N]))
        bottom_boundary = tf.square(tf.reshape(u_current[:, 0, :, :], [self.N]))
        top_boundary = tf.square(tf.reshape(u_current[:, -1, :, :], [self.N]))

        # # Define boundary loss for left, right, bottom, and top boundaries
        boundary = tf.concat([left_boundary,
                              right_boundary,
                              bottom_boundary,
                              top_boundary], axis = -1)
        boundary = tf.reduce_mean(boundary)

        # Compute final loss
        loss = self.alpha*interior + (1 - self.alpha)*boundary

        return loss