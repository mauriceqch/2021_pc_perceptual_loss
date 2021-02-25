import tensorflow as tf
import numpy as np
import scipy

# Maurice Quach: helper class for transparent usage
class NaBCE:
    def __init__(self, block_shape, m=5):
        cur_graph = tf.Graph()
        with cur_graph.as_default() as graph:
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            sess = tf.Session(config=tf_config)

            x = tf.placeholder(tf.float32, shape=[1, 1] + block_shape, name='x')
            x2 = tf.placeholder(tf.float32, shape=[1, 1] + block_shape, name='x2')
            self.naBCE = naBCE(x, x2, compute_dist_kernel(m))

            self.sess = sess
            self.cur_graph = tf.Graph()
            self.x, self.x2 = x, x2
    
    def __call__(self, x, x2):
        return self.sess.run(self.naBCE, feed_dict={self.x: x[np.newaxis, np.newaxis], self.x2: x2[np.newaxis, np.newaxis]})

# Thanks to the authors for providing their implementation
# A. Guarda, N. Rodrigues, and F. Pereira, 
# “Neighborhood Adaptive Loss Function for Deep Learning-based Point Cloud Coding with Implicit and Explicit Quantization,”
# IEEE MultiMedia, pp. 1–1, 2020, doi: 10.1109/MMUL.2020.3046691.
def naBCE(y_true, y_pred, dist_kernel):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = tf.clip_by_value(pt_1, 1e-3, .999)
    pt_0 = tf.clip_by_value(pt_0, 1e-3, .999)
    
    # Compute weights for '1' voxels, normalize and clip
    dist_weights_1 = tf.nn.conv3d(y_true, dist_kernel, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NCDHW')
    dist_weights_1 = 1 - (dist_weights_1 / tf.reduce_max(dist_weights_1))
    dist_weights_1 = tf.clip_by_value(dist_weights_1, 1e-3, 1.0)
    # Compute weights for '0' voxels, normalize and clip
    dist_weights_0 = tf.nn.conv3d(1 - y_true, dist_kernel, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NCDHW')
    dist_weights_0 = 1 - (dist_weights_0 / tf.reduce_max(dist_weights_0))    
    dist_weights_0 = tf.clip_by_value(dist_weights_0, 1e-3, 1.0)
    
    return -tf.reduce_sum(dist_weights_1 * tf.log(pt_1)) - tf.reduce_sum(dist_weights_0 * tf.log(1. - pt_0))

def compute_dist_kernel(m=5):
    # Build distance filter kernel (each filter coefficient corresponds to the inverse of the squared euclidean distance to the center of the window)
    middle_window = np.ceil(m/2).astype(int)
    full_enum = np.arange(1, m+1, dtype=np.int32)
    coor_grid = np.meshgrid(full_enum, full_enum, full_enum, indexing='ij')
    coor_list = np.transpose(np.reshape(coor_grid, [3, m*m*m]))
    dist_list = scipy.spatial.distance.cdist(coor_list, np.atleast_2d([middle_window, middle_window, middle_window]), 'sqeuclidean')
    
    dist_mat = np.reshape(np.divide(1, dist_list), [m, m, m])
    dist_mat[middle_window - 1, middle_window - 1, middle_window - 1] = 0

    dist_kernel = np.expand_dims(dist_mat, -1)
    dist_kernel = np.expand_dims(dist_kernel, -1)
    
    return dist_kernel
