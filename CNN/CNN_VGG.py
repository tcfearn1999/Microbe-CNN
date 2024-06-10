from DataLoader import CustomDataLoader
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import plot_model
import tensorflow as tf


class PrintAccuracy(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("Epoch:", epoch+1, "- Train Accuracy:", logs['accuracy'])


class VGG19_TwoInputs_Classifier:
    def __init__(self, num_classes, epochs):
        self.input_shape = (224, 224, 3)
        self.num_classes = num_classes
        self.epochs = epochs
        self.model = self.build_model()

    def build_model(self):
        resident = Input(shape=self.input_shape)
        donor = Input(shape=self.input_shape)

        # First Input CNN
        x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(resident)
        x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
        x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)

        x1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
        x1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
        x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)

        x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x1)
        x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x1)
        x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x1)
        x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x1)
        x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)

        x1 = Conv2D(512, (3, 3), activation='relu', padding='same')(x1)
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same')(x1)
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same')(x1)
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same')(x1)
        x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)

        x1 = Conv2D(512, (3, 3), activation='relu', padding='same')(x1)
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same')(x1)
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same')(x1)
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same')(x1)
        x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)

        # Second Input CNN
        x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(donor)
        x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
        x2 = MaxPooling2D((2, 2), strides=(2, 2))(x2)

        x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
        x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
        x2 = MaxPooling2D((2, 2), strides=(2, 2))(x2)

        x2 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
        x2 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
        x2 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
        x2 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
        x2 = MaxPooling2D((2, 2), strides=(2, 2))(x2)

        x2 = Conv2D(512, (3, 3), activation='relu', padding='same')(x2)
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same')(x2)
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same')(x2)
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same')(x2)
        x2 = MaxPooling2D((2, 2), strides=(2, 2))(x2)

        x2 = Conv2D(512, (3, 3), activation='relu', padding='same')(x2)
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same')(x2)
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same')(x2)
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same')(x2)

        # Concatenating Outputs
        x = Concatenate()([Flatten()(x1), Flatten()(x2)])
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        outputs = Dense(2, activation='sigmoid')(x)

        model = Model(inputs=[resident, donor], outputs=outputs)
        return model

    def build_compile(self):
        self.model = self.build_model()
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall', 'F1Score'])

    def summary(self):
        self.model.summary()

    def train(self, train_loader):
        for epoch in range(self.epochs):
            for batch in train_loader:
                train_resident = batch['resident_file']
                train_donor = batch['donor_file']
                train_labels = batch['label']

                train_labels = tf.expand_dims(train_labels, axis=1)  # Expand dimension

                self.model.fit([train_resident, train_donor],
                                train_labels,
                                epochs=1,callbacks=[PrintAccuracy()])

    def evaluate(self, test_loader, batch_size):
        for batch in test_loader:
            test_resident = batch['resident_file']
            test_donor = batch['donor_file']
            test_labels = batch['label']

            test_labels = tf.expand_dims(test_labels, axis=1)  # Expand dimension

            self.model.evaluate(x = [test_resident, test_donor],
                                y = test_labels)

    def plot(self):
        plot_model(self.model, "multi_input_model.png", show_shapes=True)

    def save(self,filename):
        self.model.save(f"./{filename}.keras")




# Define your custom data loaders for train and test sets
train_csv = "put_filepath_for_train_csv_files"
batch_size = "put_batch_size"
shuffle = True
seed = 67
test_dataloader = CustomDataLoader(train_csv, batch_size, shuffle, seed)

test_csv = "put_filepath_for_test_csv_files"
train_dataloader = CustomDataLoader(test_csv, batch_size, shuffle, seed)

epochs = "put_number_of_epochs"
num_classes = 2  # Number of output classes
model = VGG19_TwoInputs_Classifier(num_classes, epochs=epochs)

# Compile the model
model.build_compile()

# Train the model
model.train(train_loader=train_dataloader)

# Evaluate model with testing dataset
model.evaluate(test_loader=test_dataloader, batch_size = batch_size)

# Save model
model.save(filename="put_file_name.keras")




