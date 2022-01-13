import tensorflow.keras.backend as K
import tensorflow as tf


def MeanAbsoErr(y_true, y_pred):
    """
    Mean absolute error regression loss.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean absolute error
    """
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def MeanSqrtErr(y_true, y_pred):
    """
    Mean squared error regression loss.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean squared error
    """
    return K.mean(K.square(y_pred - y_true), axis=-1)


def MeanAbsPercErr(y_true, y_pred):
    """
    Mean absolute percentage error regression loss.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean absolute percentage error
    """
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    return 100. * K.mean(diff, axis=-1)


def MeanSqrtLogErr(y_true, y_pred):
    """
    Mean squared logarithmic error regression loss.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: mean squared logarithmic error
    """
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def R2CoefScore(y_true, y_pred):
    """
    :math:`R^2` (coefficient of determination) regression score function.

    Best possible score is 1.0, lower values are worse.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: R2    
    """
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    SS_tot = tf.reduce_sum(
        tf.square(y_true - tf.reduce_mean(y_true, axis=-1)), axis=-1)
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))


def AdjR2CoefScore(y_true, y_pred):
    """
    Adjusted R2 regression score function with default inputs.

    Best possible score is 1.0, lower values are worse.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: adjusted R2
    """
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    SS_tot = tf.reduce_sum(
        tf.square(y_true - tf.reduce_mean(y_true, axis=-1)), axis=-1)
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon())) * (1 - (1 - R2CoefScore(y_true, y_pred)) * (tf.cast(tf.size(y_true), tf.float32) - 1) / (tf.cast(tf.size(y_true), tf.float32) - tf.cast(tf.rank(y_true), tf.float32) - 1))
    # SS_res =  tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    # SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=-1)), axis=-1)
    # adj_SS_res = tf.cast(SS_res / (K.shape(y_true)[0] - 1), tf.int32)
    # adj_SS_tot = tf.cast(SS_tot / (K.shape(y_true)[0] - 1), tf.int32)
    # return (1 - adj_SS_res/(adj_SS_tot + tf.keras.backend.epsilon()))


def RootMeanSqrtLogErr(y_true, y_pred):
    """
    Root Mean Squared Logarithm Error
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: root mean squared logarithm error
    """
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def RootMeanSqrtErr(y_true, y_pred):
    """
    Root Mean Squared Error
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: root mean squared error
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def SymMeanAbsPercErr(y_true, y_pred):
    """
    Symmetric mean absolute percentage error regression loss.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: symmetric mean absolute percentage error
    """
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    return 100. * K.mean(K.mean(diff, axis=-1))


def SymMeanAbsPercLogErr(y_true, y_pred):
    """
    Symmetric mean absolute percentage log error regression loss.

    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: symmetric mean absolute percentage error
    """
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    return K.log(K.mean(K.mean(diff, axis=-1)))


def NormRootMeanSqrtErr(y_true, y_pred):
    """
    Normalized Root Mean Squared Error
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples

    Returns:
        [float]: normalized root mean squared error
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) / K.mean(K.abs(y_true), axis=-1)
