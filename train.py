"""
Train our LSTM on extracted features.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
from extract_features import extract_features
import time
import os.path
import sys
import matplotlib.pyplot as plt
import os
def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    #tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    #if load_to_memory:
        # Get data.
    X, y = data.get_all_sequences_in_memory('train', data_type)
    X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    #else:
        # Get generators.
    generator = data.frame_generator(batch_size, 'train', data_type)
    val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[ early_stopper, csv_logger, checkpointer],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        history = rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[ early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=4)
        score, acc1 = rm.model.evaluate(X_test, y_test, batch_size=batch_size)
        print('Test score:', score*100)
        print('Test accuracy:', acc1*100)
        # Plot training & validation accuracy values
        

def main():
    """These are the main training settings. Set each before running
    this file."""
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


    if (len(sys.argv) == 5):
        seq_length = int(sys.argv[1])
        class_limit = int(sys.argv[2])
        image_height = int(sys.argv[3])
        image_width = int(sys.argv[4])
    else:
        print ("Usage: python train.py sequence_length class_limit image_height image_width")
        print ("Example: python train.py 75 2 720 1280")
        exit (1)

    sequences_dir = os.path.join('data', 'sequences')
    if not os.path.exists(sequences_dir):
        os.mkdir(sequences_dir)

    checkpoints_dir = os.path.join('data', 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    # model can be only 'lstm'
    model = 'lstm'
    saved_model = None  # None or weights file
    load_to_memory = False # pre-load the sequences into memory
    batch_size = 16
    nb_epoch = 50
    data_type = 'features'
    image_shape = (image_height, image_width, 3)

    extract_features(seq_length=seq_length, class_limit=class_limit, image_shape=image_shape)
    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
