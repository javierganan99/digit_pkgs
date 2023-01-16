import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

# Needed to asign memory to GPUs ###############################################
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
################################################################################


def CNN_digit():
    """
    GRADED:
    Define a funcion called CNN_digit() that defines and return a CNN model.
    -- Use the Sequential Class from keras: https://keras.io/api/models/sequential/
    -- Use the add method to add the layers of the CNN
    """
    model = tf.keras.Sequential()
    
    # Add 2 Conv layer with 32 filters, 5x5 kernel and same padding
    # ~ 2 lines

    # Add a maxpooling layer with 2x2 kernel and no stride
    # ~ 1 line

    # Add 2 Conv layer with 64 filters, 3x3 kernel and same padding
    # ~ 2 lines

    # Add a maxpooling layer with 2x2 kernel and stride 2
    # ~ 1 line

    # Add dropout with probability 0.25
    # ~ 1 line

    # Flatten the previous layer
    # ~ 1 line

    # Add a dense layer with 256 neurons and relu activation
    # ~ 1 line

    # Add dropout with probability 0.25
    # ~ 1 line

    # Add a dense layer with 10 output neurons and softmax activation
    # ~ 1 line

    return model

if __name__== '__main__':
    IMG_SIZE = 28
    IMG_DIR = './../dataset'
    BATCH_SIZE = 8
    
    def min_max_norm(x):
        x = (x - x.min()) / (x.max() - x.min())
        return x 
    
    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2,
        preprocessing_function = min_max_norm)
    
    train_gen=datagen.flow_from_directory(IMG_DIR, target_size=(IMG_SIZE,IMG_SIZE), color_mode='grayscale',class_mode='categorical',seed=1,
                                batch_size=BATCH_SIZE, subset='training', shuffle = True)
    
    valid_gen=datagen.flow_from_directory(IMG_DIR, target_size=(IMG_SIZE,IMG_SIZE), color_mode='grayscale',class_mode='categorical',seed=1,
                                batch_size=BATCH_SIZE, subset='validation')
    
    # Define the model
    model = CNN_digit()
    
    # Compile model
    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    # Model training
    """
    GRADED:
    Use fit_generator() method of the model to train it. 
    -- Use generator argument to pass the training data
    -- steps per epoch = train_gen.n // train_gen.batch_size
    -- Use epochs = 100
    -- Use validation_data argument to pass the validation data
    """
    
    # Save model
    model.save('./../model/digit_classification.h5')




    
