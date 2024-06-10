from DataLoader import CustomDataLoader
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import plot_model
import tensorflow as tf

class PrintAccuracy(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("Epoch:", epoch+1, "- Train Accuracy:", logs['accuracy'])

class ResNet50_TwoInputs_Classifier:
    def __init__(self, num_classes, epochs):
        self.input_shape = (224, 224, 3)
        self.num_classes = num_classes
        self.epochs = epochs
        self.model = self.build_model()

    def build_model(self):
        #using pre-built model to reduce energy consumption for initial tests
        base_model = ResNet50(weights='imagenet', include_top=False)

        resident_input = Input(shape=self.input_shape, name='resident_input')
        donor_input = Input(shape=self.input_shape, name='donor_input')

        resident_features = base_model(resident_input)
        donor_features = base_model(donor_input)

        x = Concatenate()([Flatten()(resident_features), Flatten()(donor_features)])
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[resident_input, donor_input], outputs=outputs)
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

                train_labels = tf.expand_dims(train_labels, axis=1)

                self.model.fit([train_resident, train_donor],
                                train_labels,
                                epochs=1, callbacks=[PrintAccuracy()])

    def evaluate(self, test_loader):
        for batch in test_loader:
            test_resident = batch['resident_file']
            test_donor = batch['donor_file']
            test_labels = batch['label']

            test_labels = tf.expand_dims(test_labels, axis=1)

            self.model.evaluate(x = [test_resident, test_donor],
                                y = test_labels)

    def plot(self):
        plot_model(self.model, "Test.png", show_shapes=True)

    def save(self, filename):
        self.model.save(f"./{filename}")

# Define your custom data loaders for train and test sets
train_csv = "put_filepath_for_train_csv_files"
batch_size = "put_filepath_for_batchsize"
shuffle = True
seed = 2    #for reproducabilty
train_dataloader = CustomDataLoader(train_csv, batch_size, shuffle, seed)

test_csv = "put_filepath_for_train_csv_files"
test_dataloader = CustomDataLoader(test_csv, batch_size, shuffle, seed)

num_classes = 2  
epochs = "put_number_of_epochs"
model = ResNet50_TwoInputs_Classifier(num_classes, epochs=epochs)

# Compile the model
model.build_compile()

#plot model
model.plot()

# Train the model
model.train(train_loader=train_dataloader)

# Evaluate model with testing dataset
model.evaluate(test_loader=test_dataloader)

# Save model
model.save(filename="put_file_name.keras")

