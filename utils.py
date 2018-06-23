import keras as K
from keras.utils import conv_utils
from keras.layers.convolutional import UpSampling3D
from keras.engine import InputSpec
from tensorlayer.layers import *


class UpSampling3D(Layer):
    def __init__(self, size=(2, 2, 2), **kwargs):
        self.size = conv_utils.normalize_tuple(size, 3, 'size')
        self.input_spec = InputSpec(ndim=5)
        super(UpSampling3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        dim1 = self.size[0] * input_shape[1] if input_shape[1] is not None else None
        dim2 = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        dim3 = self.size[2] * input_shape[3] if input_shape[3] is not None else None
        return (input_shape[0],
                dim1,
                dim2,
                dim3,
                input_shape[4])

    def call(self, inputs):
        return K.resize_volumes(inputs,
                                self.size[0], self.size[1], self.size[2],
                                self.data_format)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(UpSampling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def smooth_gan_labels(y):
    if y == 0:
        y_out = tf.random_uniform(shape=y.get_shape(), minval=0.0, maxval=0.3)
    else:
        y_out = tf.random_uniform(shape=y.get_shape(), minval=0.7, maxval=1.2)

    return y_out


def subPixelConv3d(net, img_width, img_height, img_depth, stepsToEnd, n_out_channel):
    i = net
    r = 2
    a, b, z, c = int(img_width / (2 * stepsToEnd)), int(img_height / (2 * stepsToEnd)), int(
        img_depth / (2 * stepsToEnd)), tf.shape(i)[3]
    bsize = tf.shape(i)[0]  # Handling Dimension(None) type for undefined batch dim
    xs = tf.split(i, r, 4)  # b*h*w*d*r*r*r
    xr = tf.concat(xs, 3)  # b*h*w*(r*d)*r*r
    xss = tf.split(xr, r, 4)  # b*h*w*(r*d)*r*r
    xrr = tf.concat(xss, 2)  # b*h*(r*w)*(r*d)*r
    x = tf.reshape(xrr, (bsize, r * a, r * b, r * z, n_out_channel))  # b*(r*h)*(r*w)*(r*d)*n_out n_out=64/2^

    return x


def aggregate(patches):
    margin = 16
    volume = np.empty([224, 224, 152, 1])
    volume[0:112, 0:112, 0:76, :] = patches[0, 0:112, 0:112, 0:76, :]
    volume[0:112, 0:112, 76:, :] = patches[1, 0:112, 0:112, margin:, :]
    volume[0:112, 112:, 0:76, :] = patches[2, 0:112, margin:, 0:76, :]
    volume[0:112, 112:, 76:, :] = patches[3, 0:112, margin:, margin:, :]
    volume[112:, 0:112, 0:76, :] = patches[4, margin:, 0:112, 0:76, :]
    volume[112:, 0:112, 76:, :] = patches[5, margin:, 0:112, margin:, :]
    volume[112:, 112:, 0:76, :] = patches[6, margin:, margin:, 0:76, :]
    volume[112:, 112:, 76:, :] = patches[7, margin:, margin:, margin:, :]
    return volume


def aggregate2(patches):
    margin = 8
    volume = np.empty([112, 112, 76, 1])
    volume[0:56, 0:56, 0:38, :] = patches[0, 0:56, 0:56, 0:38, :]
    volume[0:56, 0:56, 38:, :] = patches[1, 0:56, 0:56, margin:, :]
    volume[0:56, 56:, 0:38, :] = patches[2, 0:56, margin:, 0:38, :]
    volume[0:56, 56:, 38:, :] = patches[3, 0:56, margin:, margin:, :]
    volume[56:, 0:56, 0:38, :] = patches[4, margin:, 0:56, 0:38, :]
    volume[56:, 0:56, 38:, :] = patches[5, margin:, 0:56, margin:, :]
    volume[56:, 56:, 0:38, :] = patches[6, margin:, margin:, 0:38, :]
    volume[56:, 56:, 38:, :] = patches[7, margin:, margin:, margin:, :]
    return volume
