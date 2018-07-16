import pickle
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.models import Model
from keras.layers import Input, Flatten, Dense

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_string('batch_size', 128, "Batch size, default is 128")
flags.DEFINE_string('epochs', 1, "Epochs, default is 1")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    print("Train size: ", len(X_train))
    print("validation size: ", len(X_val))
    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    n_class = len(np.unique(y_train))
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    input_shape = X_train.shape[1:]
    print("input shape: ", input_shape)
    input = Input(shape = input_shape)
    x = Flatten()(input)
    output = Dense(n_class, activation="softmax")(x)
    model = Model(inputs=input, outputs=output)
    # TODO: train your model here
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_date=(X_val, y_val), shuffle=True)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
