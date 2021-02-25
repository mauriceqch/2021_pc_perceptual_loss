import tensorflow as tf
import os
from tensorflow.keras.layers import Layer, Conv3D, Conv3DTranspose

# Helper class for transparent usage
class PerceptualLoss:
    def __init__(self, checkpoint_dir, block_shape, filters=32):
        cur_graph = tf.Graph()
        with cur_graph.as_default() as graph:
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            sess = tf.Session(config=tf_config)

            model = PerceptualModel(filters=filters)
            x = tf.placeholder(tf.float32, shape=[1, 1] + block_shape, name='x')
            x2 = tf.placeholder(tf.float32, shape=[1, 1] + block_shape, name='x2')
            x_tilde = model(x)
            x_tilde2 = model(x2)
            graph = tf.get_default_graph()
            model_name = model.name
            tensors = {
                'x': x,
                'x1': graph.get_tensor_by_name(f'{model_name}/conv3d_0/Relu:0'),
                'x2': graph.get_tensor_by_name(f'{model_name}/conv3d_1/Relu:0'),
                'y0': graph.get_tensor_by_name(f'{model_name}/conv3d_2/Relu:0'),
                'y1': graph.get_tensor_by_name(f'{model_name}/conv3dt_0/Relu:0'),
                'y2': graph.get_tensor_by_name(f'{model_name}/conv3dt_1/Relu:0'),
                'x_tilde': graph.get_tensor_by_name(f'{model_name}/conv3dt_2/Sigmoid:0'),
            }
            tensors2 = {
                'x': x2,
                'x1': graph.get_tensor_by_name(f'{model_name}/conv3d_0/Relu:0'),
                'x2': graph.get_tensor_by_name(f'{model_name}_1/conv3d_1/Relu:0'),
                'y0': graph.get_tensor_by_name(f'{model_name}_1/conv3d_2/Relu:0'),
                'y1': graph.get_tensor_by_name(f'{model_name}_1/conv3dt_0/Relu:0'),
                'y2': graph.get_tensor_by_name(f'{model_name}_1/conv3dt_1/Relu:0'),
                'x_tilde': graph.get_tensor_by_name(f'{model_name}_1/conv3dt_2/Sigmoid:0'),
            }
            # Assumes batch_size=1
            y0_mse = tf.reduce_mean(tf.square(tensors['y0'] - tensors2['y0']))
            # Squeeze batch_size and average over features
            y0_features_mse = tf.reduce_mean(tf.reshape(tf.squeeze(tf.square(tensors['y0'] - tensors2['y0']), axis=0), (filters, -1)), axis=1)

            saver = tf.train.Saver(save_relative_paths=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
            checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            print(f'Restoring checkpoint {checkpoint}')
            saver.restore(sess, checkpoint)

            self.x, self.x2, self.y0_mse, self.y0_features_mse, self.tensors, self.tensors2 = x, x2, y0_mse, y0_features_mse, tensors, tensors2
            self.sess = sess
            self.cur_graph = tf.Graph()
            self.filters = filters
    
    def __call__(self, x, x2, features=False):
        if features:
            tensor = self.y0_features_mse
        else:
            tensor = self.y0_mse
        return self.sess.run(tensor, feed_dict={self.x: x[np.newaxis, np.newaxis], self.x2: x2[np.newaxis, np.newaxis]})


# Define input pipeline
def pc_to_tf(points, dense_tensor_shape, data_format):
    x = points
    assert data_format in ['channels_last', 'channels_first']
    # Add one channel (channels_last convention)
    if data_format == 'channels_last':
        x = tf.pad(x, [[0, 0], [0, 1]])
    else:
        x = tf.pad(x, [[0, 0], [1, 0]])
    st = tf.sparse.SparseTensor(x, tf.ones_like(x[:, 0]), dense_tensor_shape)
    return st


def process_x(x, dense_tensor_shape):
    x = tf.sparse.to_dense(x, default_value=0, validate_indices=False)
    x.set_shape(dense_tensor_shape)
    x = tf.cast(x, tf.float32)
    return x


def input_fn(points, batch_size, dense_tensor_shape, data_format, repeat=True, shuffle=True, prefetch_size=1):
    # Create input data pipeline.
    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_generator(lambda: iter(points), tf.int64, tf.TensorShape([None, 3]))
        if shuffle:
            dataset = dataset.shuffle(len(points))
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.map(lambda x: pc_to_tf(x, dense_tensor_shape, data_format))
        dataset = dataset.map(lambda x: process_x(x, dense_tensor_shape))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_size)

    return dataset

# Define model
class PerceptualModel(Layer):
    def __init__(self, filters=32, data_format='channels_first', activation=tf.nn.relu, **kwargs):
        super().__init__()
        params = {'strides': (2, 2, 2), 'padding': 'same', 'data_format': data_format, 'activation': activation, 'use_bias': True}
        self.conv3d_0 = Conv3D(name='conv3d_0', filters=filters, kernel_size=(5, 5, 5), **params)
        self.conv3d_1 = Conv3D(name='conv3d_1', filters=filters, kernel_size=(5, 5, 5), **params)
        self.conv3d_2 = Conv3D(name='conv3d_2', filters=filters, kernel_size=(5, 5, 5), **params)
        self.conv3dt_0 = Conv3DTranspose(name='conv3dt_0', filters=filters, kernel_size=(5, 5, 5), **params)
        self.conv3dt_1 = Conv3DTranspose(name='conv3dt_1', filters=filters, kernel_size=(5, 5, 5), **params)
        self.conv3dt_2 = Conv3DTranspose(name='conv3dt_2', filters=1, kernel_size=(5, 5, 5), **{**params, 'activation': 'sigmoid'})

    def call(self, x):
        x0 = x
        x1 = self.conv3d_0(x0)
        x2 = self.conv3d_1(x1)
        y0 = self.conv3d_2(x2)
        y1 = self.conv3dt_0(y0)
        y2 = self.conv3dt_1(y1)
        x_tilde = self.conv3dt_2(y2)

        return x_tilde