import tensorflow as tf

class TFDataset(tf.data.Dataset):
    def __init__(self, datasamples, is_inference: bool):
        self.x = datasamples["images"] / 255. # new datasamples with normalized datas
        self.y = datasamples["labels"]
        self.is_inference = is_inference
        self.nb_classes = 8 # labels go from 0 to 7
        self.one_hots = tf.one_hot(indices=list(range(self.nb_classes)), depth=self.nb_classes, dtype='float32')
        # =
        # [[1., 0., 0., 0., 0., 0., 0., 0.],
        #  [0., 1., 0., 0., 0., 0., 0., 0.],
        #  [0., 0., 1., 0., 0., 0., 0., 0.],
        #  ...,
        # ]
        self.input_shape = self.x[0].shape # shape of input images (150,150,3)


    def __getitem__(self, idx):

        if self.is_inference:
            x = tf.convert_to_tensor(value=self.x[idx][None, ...], dtype='float64') # keep float64
            return x

        else:
            x = tf.convert_to_tensor(value=self.x[idx][None, ...], dtype='float64')
            y = tf.expand_dims(self.one_hots[self.y[idx]], axis = 0)
            return x, y
    
    def __len__(self):
        return len(self.x)   

    # adding missing methods

    def _inputs(self):
        """Returns a list of the input datasets of the dataset."""
        if self.is_inference:
            return []
        else:
            return tf.TensorSpec(shape=self.input_shape, dtype=tf.float64), tf.TensorSpec(shape=(self.nb_classes,), dtype=tf.float32)

    def element_spec(self):
        """The type specification of an element of this dataset."""
        if self.is_inference:
            return tf.TensorSpec(shape=self.input_shape, dtype=tf.float64)
        else:
            return tf.TensorSpec(shape=self.input_shape, dtype=tf.float64), tf.TensorSpec(shape=(self.nb_classes,), dtype=tf.float32)