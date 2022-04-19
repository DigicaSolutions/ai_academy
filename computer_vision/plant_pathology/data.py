import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


def load_image(
    path, image_size, num_channels, interpolation, crop_to_aspect_ratio=False
):
    """Load an image from a path and resize it."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=num_channels, expand_animations=False)
    if crop_to_aspect_ratio:
        img = tf.keras.preprocessing.image.smart_resize(
            img, image_size, interpolation=interpolation
        )
    else:
        img = tf.image.resize(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img


def create_datasets(images_path, images_csv, used_classes=None, random_state=None):
    images_df = pd.read_csv(images_csv)
    if not images_path.endswith("/"):
        images_path += "/"
    images_df["image"] = images_path + images_df["image"]

    label_encoder = LabelEncoder()

    if used_classes is not None:
        images_df = images_df.query("labels in @used_classes")
    labels = label_encoder.fit_transform(images_df["labels"])

    x_train, x_val, y_train, y_val = train_test_split(
        images_df["image"],
        labels,
        stratify=labels,
        test_size=0.2,
        random_state=random_state,
    )

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    return train_ds, val_ds, label_encoder
