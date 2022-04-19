import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import compute_class_weight
import tensorflow as tf

from plant_pathology.confusion_matrix import ConfusionMatrixLogger
from plant_pathology.data import load_image, create_datasets


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("images_path")
    arg_parser.add_argument("annotations_path")
    args = arg_parser.parse_args()

    train_model(args.images_path, args.annotations_path)


def create_model(input_shape, n_outputs):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.RandomFlip(input_shape=input_shape),
            tf.keras.layers.RandomRotation(0.5),
            tf.keras.layers.Normalization(
                axis=-1,
                mean=[0.47556898, 0.6262431, 0.3969909],
                variance=[0.0282571, 0.02152061, 0.03094865],
            ),
            tf.keras.layers.Conv2D(32, 11, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(16, 5, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1000, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(n_outputs),
        ]
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        if n_outputs == 1
        else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        optimizer="adam",
    )
    return model


def gen_to_ds(ds_gen):
    return tf.data.Dataset.from_generator(
        ds_gen,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                tf.TensorSpec(shape=tuple(), dtype=tf.uint8),
            )
        ),
    )


def preprocess_img(
    image_path, label, img_shape=(224, 224, 3), crop_to_aspect_ratio=True
):
    height, width, n_channels = img_shape
    img = load_image(
        image_path,
        (height, width),
        n_channels,
        "bilinear",
        crop_to_aspect_ratio=crop_to_aspect_ratio,
    )
    img = img / 255
    return img, label


def train_model(images_path, annotations_path):
    used_classes = [
        "scab",
        "healthy",
        "frog_eye_leaf_spot",
        "rust",
        "complex",
        "powdery_mildew",
    ]
    input_shape = (300, 300, 3)
    train_ds, val_ds, label_encoder = create_datasets(
        images_path, annotations_path, used_classes, random_state=42
    )
    train_ds = (
        train_ds.shuffle(18632)
        .map(lambda x, y: preprocess_img(x, y, input_shape))
        .batch(128)
    )
    val_ds = val_ds.map(lambda x, y: preprocess_img(x, y, input_shape)).batch(128)
    model = create_model(input_shape, n_outputs=len(used_classes))

    # Callbacks
    log_dir = pathlib.Path("logs-6-classes/8-bigger")
    tb_callback = tf.keras.callbacks.TensorBoard(str(log_dir))
    cm_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=ConfusionMatrixLogger(model, val_ds, label_encoder, log_dir)
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(log_dir / "model")
    callbacks = [tb_callback, cm_callback, checkpoint_callback]
    ###########
    y_train = [y.numpy() for _, ys in train_ds for y in ys]
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1000,
        callbacks=callbacks,
        class_weight=dict(enumerate(class_weights)),
    )

    pred = model.predict(val_ds)
    pred = pred.argmax(-1)
    y = [y.numpy() for _, y_batch in val_ds for y in y_batch]
    print(f'F1 score: {f1_score(y, pred, average="macro")}')
    print(confusion_matrix(y, pred))


if __name__ == "__main__":
    main()
