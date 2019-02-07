import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

tf.enable_eager_execution()

def linear_dataset(beta_true, N, noise_std=0.1):
    """
    For now just assuming that beta_true is a scalar
    """
    X = tf.random_normal(shape=[N, 1], mean=0, stddev=1)
    noise = tf.random_normal(shape=[N, 1], stddev=noise_std)

    Y = beta_true * X + noise

    return X, Y

# scale is the std div
beta = tfd.Normal(loc=0, scale=20)

def trainable_normal(name=None):
    with tf.variable_scope(None, default_name="trainable_normal"):
        return tfd.Normal(loc=tf.get_variable("loc"), scale=tf.get_variable("scale"), name=name)
