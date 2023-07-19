import tensorflow as tf

# Following data_loader is not used

def tf_dataloader(dataset, batch_size=1, shuffle=False) -> tf.data.Dataset:
    # Create a tf.data.Dataset object from the dataset
    data = tf.data.Dataset.from_tensor_slices(dataset)
    
    if shuffle:
        # Shuffle the elements of the dataset
        data = data.shuffle(buffer_size=len(dataset))
    
    # Group the elements into batches
    data = data.batch(batch_size)
    
    return data