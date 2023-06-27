import tensorflow as tf
#import TFDatasetFile
from tensorflow_algorithms import TFDatasetFile


def tf_dataloader(dataset, batch_size , shuffle=False) -> tf.data.Dataset:
    #Create a dataset divided in batchs 
    if isinstance(dataset,TFDatasetFile.TFDataset) :

        def sequential_access_dataset():
            for idx in range(len(dataset)):
                yield dataset[idx]
    
        data = tf.data.Dataset.from_generator(sequential_access_dataset,(tf.float32,tf.float32))
        data = data.batch(batch_size)
       
    else :
        data = dataset.batch(batch_size) 
    
    
    if shuffle:
        # Shuffle the elements of the dataset
        data = data.shuffle(buffer_size=len(dataset))
   
    return data