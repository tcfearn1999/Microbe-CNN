import numpy as np
import pandas as pd
import tensorflow as tf

class CustomDataLoader:
    def __init__(self, csv_file, batch_size, shuffle=True, seed=None):
        self.data = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.idx = 0
        self._shuffle_data()

    def _load_npy_file(self, file_path):
        return np.load(file_path)

    def _shuffle_data(self):
        if self.shuffle and self.seed is not None:
            np.random.seed(self.seed)
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def __iter__(self):
        self._shuffle_data()
        self.idx = 0
        return self

    def __len__(self):
        return len(self.data) // self.batch_size + 1

    def __next__(self):
        if self.idx >= len(self.data):
            raise StopIteration

        batch_data = self.data.iloc[self.idx:self.idx+self.batch_size]
        batch_dict = self._load_batch_data(batch_data)
        self.idx += self.batch_size
        return batch_dict

    def _load_batch_data(self, batch_data):
        batch_dict = {'resident_file': [], 'donor_file': [], 'label': []}

        for idx, row in batch_data.iterrows():
            resident_file = tf.convert_to_tensor(self._load_npy_file(row['resident_input']))
            donor_file = tf.convert_to_tensor(self._load_npy_file(row['donor_output']))

            batch_dict['resident_file'].append(resident_file)
            batch_dict['donor_file'].append(donor_file)
            batch_dict['label'].append(row['stoch_antag'])  # Use label from the row

        batch_dict['resident_file'] = tf.stack(batch_dict['resident_file'])
        batch_dict['donor_file'] = tf.stack(batch_dict['donor_file'])
        batch_dict['label'] = tf.convert_to_tensor(batch_dict['label'])

       	return batch_dict

