import tensorflow as tf

class DataLoader:
    def __init__(self, img_dir, img_size=(64, 64), batch_size=128, split=0.8):
        self.img_dir = img_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.img_data = None
        self.total = None
        self.split = split

    def load_data(self, num_images=None):
        # Load the images
        img_data = tf.data.Dataset.list_files(self.img_dir + '*jpg')
        print("Loading Data")
        if num_images is not None:
            img_data = img_data.take(num_images)
        img_data = img_data.map(lambda x: tf.io.read_file(x))
        img_data = img_data.map(lambda x: tf.image.decode_jpeg(x, channels=3))
        img_data = img_data.map(lambda x: tf.image.resize(x, self.img_size))
        img_data = img_data.map(lambda x: (x / 127.5) - 1.0)

        # Save the total number of images in the dataset
        self.total = tf.data.experimental.cardinality(img_data).numpy()

        # Split the dataset into train and validation sets
        train_dataset = img_data.take(int(self.total * self.split))
        val_dataset = img_data.skip(int(self.total * self.split))

        # Shuffle and batch the datasets
        self.img_data = {
            'train': train_dataset.shuffle(buffer_size=10000).batch(batch_size=self.batch_size, drop_remainder=True).prefetch(buffer_size=10),
            'val': val_dataset.shuffle(buffer_size=10000).batch(batch_size=self.batch_size, drop_remainder=True).prefetch(buffer_size=10)
        }

        # Print the size of the datasets
        print("Finished Loading")
        print(f"Train dataset size: {tf.data.experimental.cardinality(train_dataset).numpy()}")
        print(f"Validation dataset size: {tf.data.experimental.cardinality(val_dataset).numpy()}\n\n")

    def get_data(self):
        return self.img_data

    def get_total(self):
        return self.total