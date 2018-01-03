import numpy as np
import librosa
import platform
import os
import glob
from numpy.random import shuffle
import pescador
from sklearn.preprocessing import MinMaxScaler
# for working on both linux and windows
dir_path = os.path.dirname(os.path.realpath(__file__))
seperator = '/'
if platform.system() == 'Windows':
    seperator = '\\'


class DataHandler(object):
    def __init__(self, dataset='ESC-US',
                 bands=60,
                 hop_length=1024,
                 window=1024,
                 use_delta=False,
                 val_size=10000,
                 test_size=10000,
                 train_size=None,
                 normalize=False,
                 label_whitelist=None):

        # Reproducability
        np.random.seed(1337)

        self.dataset = dataset

        # For ESC datasets
        self.max_value = 5
        self.min_value = -20

        if self.dataset == 'ESC-US':
            self.file_ext = "*.ogg"
            self.data_path = os.path.join(dir_path, 'datasets', 'ESC-US')
            self.data_dirs = map(lambda d: os.path.join(self.data_path, d, self.file_ext),
                            ['01', '02', '03', '04', '05'])

            self.sampling_rate = 44100
            self.max_raw = 220500
            self.labeled = False
        elif self.dataset == 'ESC-50':
            self.file_ext = "*.ogg"
            self.data_path = os.path.join(dir_path, 'datasets', 'ESC-50')
            self.esc_50_list = (['101 - Dog',
                                 '102 - Rooster',
                                 '103 - Pig',
                                 '104 - Cow',
                                 '105 - Frog',
                                 '106 - Cat',
                                 '107 - Hen',
                                 '108 - Insects',
                                 '109 - Sheep',
                                 '110 - Crow',
                                 '201 - Rain',
                                 '202 - Sea waves',
                                 '203 - Crackling fire',
                                 '204 - Crickets',
                                 '205 - Chirping birds',
                                 '206 - Water drops',
                                 '207 - Wind',
                                 '208 - Pouring water',
                                 '209 - Toilet flush',
                                 '210 - Thunderstorm',
                                 "301 - Crying baby",
                                 "302 - Sneezing",
                                 "303 - Clapping",
                                 "304 - Breathing",
                                 "305 - Coughing",
                                 "306 - Footsteps",
                                 "307 - Laughing",
                                 "308 - Brushing teeth",
                                 "309 - Snoring",
                                 "310 - Drinking - sipping",
                                 "401 - Door knock",
                                 "402 - Mouse click",
                                 "403 - Keyboard typing",
                                 "404 - Door - wood creaks",
                                 "405 - Can opening",
                                 "406 - Washing machine",
                                 "407 - Vacuum cleaner",
                                 "408 - Clock alarm",
                                 "409 - Clock tick",
                                 "410 - Glass breaking",
                                 "501 - Helicopter",
                                 "502 - Chainsaw",
                                 "503 - Siren",
                                 "504 - Car horn",
                                 "505 - Engine",
                                 "506 - Train",
                                 "507 - Church bells",
                                 "508 - Airplane",
                                 "509 - Fireworks",
                                 "510 - Hand saw"
                                 ])
            if label_whitelist is not None:
                self.esc_50_list = list(filter(lambda x: x in label_whitelist, self.esc_50_list))

            values = np.arange(50)
            self.label_dict = dict(zip(self.esc_50_list, values))

            self.data_dirs = map(lambda d: os.path.join(self.data_path, d, self.file_ext), self.esc_50_list)

            self.sampling_rate = 44100
            self.max_raw = 220500
            self.labeled = True

        elif self.dataset == 'UrbanSound8K':
            self.file_ext = "*.wav"
            self.data_path = os.path.join(dir_path, 'datasets', 'UrbanSound8K', 'audio')
            self.data_dirs = map(lambda d: os.path.join(self.data_path, d, self.file_ext),
                            ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6',
                             'fold7', 'fold8', 'fold9', 'fold10'])
            self.labeled = True
            self.sampling_rate = 22050
            self.max_raw = 88200
            self.max_value = 0
            self.min_value = 0

        else:
            print("Dataset not implemented in data_streamer")

        if self.dataset != 'ESC-50' and label_whitelist is not None:
            raise ValueError("`label_whitelist` is only supported for dataset='ESC-50'")

        # Load the data
        self.test_size = test_size
        self.val_size = val_size
        self.train_size = train_size

        # Lists of files
        self.train_files = []
        self.val_files = []
        self.test_files = []

        # Loads the filenames into the above lists
        self.load_data_files()

        # Track cool stuff
        self.number_of_samples = 0
        self.batches_left_in_epoch = 0
        self.batch_size = 1
        self.number_of_train_batches = 0
        self.number_of_val_batches = 0
        self.number_of_test_batches = 0


        # Use delta features
        self.delta = use_delta

        # Values for features
        self.bands = bands
        self.hop_length = hop_length
        self.window = window

        self.normalize = False
        self.epoch_done = False



    def load_sample(self, file_name):
        sound_raw, sample_rate = librosa.load(file_name, sr=self.sampling_rate)

        # Pad zeroes to raw sound
        num_zeros = self.max_raw - len(sound_raw)

        if num_zeros > 0:
            append_array = np.zeros(num_zeros)
            sound_raw = np.append(sound_raw, append_array)
        else:
            sound_raw = sound_raw[:self.max_raw]

        # Normalize sound
        # According to its own max? or wtf
        max_abs_value = np.max(np.abs(sound_raw))

        # If all values are 0, return it
        # The encoder will hopefully learn a cluster of silence
        if max_abs_value == 0.0:
            return sound_raw, sample_rate

        normalization_factor = 1 / max_abs_value
        sound_raw = sound_raw * normalization_factor

        return sound_raw, sample_rate


    def extract_feature(self, sound_raw, sample_rate):
        """
        @return: shape = (features, bands, frames) for 5 sec
        """
        melspec = librosa.feature.melspectrogram(sound_raw,
                                                 sr=sample_rate,
                                                 n_mels=self.bands,
                                                 n_fft=self.window,
                                                 hop_length=self.hop_length)

        max_abs_value = np.max(np.abs(sound_raw))
        logspec = librosa.logamplitude(melspec, ref_power=np.max)

        if self.delta:
            delta = librosa.feature.delta(logspec)
            features = np.concatenate((np.expand_dims(logspec, 0), np.expand_dims(delta, 0)), axis=0)
        else:
            features =  np.expand_dims(logspec, 0)

        return features


    def get_sample_label(self, file_path):
        label = ''
        file_parts = file_path.split(seperator)
        if self.dataset == "UrbanSound8K":
            label = file_parts[-1].split('-')[1]
        elif self.dataset == "ESC-50":
            label = file_parts[-2]
            label = self.label_dict[label]

        return label

    def generate_sample(self, file_list):
        """
        Countinous stream of single samples from dataset
        :param file_list: List with file-names of the sound files of dataset
        :return: dict(X=x)
        """
        n = len(file_list)
        counter = 0

        while True:

            # If we have iterated through all data, shuffle and reset counter
            if counter % n == 0:
                shuffle(file_list)
                counter = 0

            file_path = file_list[counter]

            sound_raw, sr = self.load_sample(file_path)
            X = self.extract_feature(sound_raw, sr)

            # Increment
            counter += 1

            # Check if how many batches left
            if counter % self.batch_size == 0:
                self.batches_left_in_epoch = self.batches_left_in_epoch - 1
                if self.batches_left_in_epoch == 0:
                    self.batches_left_in_epoch = self.number_of_train_batches
                    self.epoch_done = True

            if (self.labeled):
                y = self.get_sample_label(file_path)
                # if self.label_whitelist is None or y in self.label_whitelist:
                yield dict(X=X, y=y)
            else:
                yield dict(X=X)


    def load_data_files(self):
        temp_files_list = []
        for dir_ in self.data_dirs:
            for fn in glob.glob(dir_):
                temp_files_list.append(fn)

        self.create_splits(temp_files_list)

    def create_splits(self, file_list):
        shuffle(file_list)
        last_test_index = self.test_size
        last_val_index = self.test_size + self.val_size

        self.test_files = file_list[0:last_test_index]
        self.val_files = file_list[last_test_index:last_val_index]

        if self.train_size:
            self.train_files = file_list[last_val_index:last_val_index + self.train_size] 
        else:
            self.train_files = file_list[last_val_index:]
            self.train_size = len(self.train_files)

        print("Training dataset size: {0}".format(len(self.train_files)))
        print("Validation dataset size: {0}".format(len(self.val_files)))
        print("Testing dataset size: {0}".format(len(self.test_files)))


    def get_streamer(self, data_split):
        if data_split == "train":
            streamer = self.generate_sample(self.train_files)
            return pescador.Streamer(streamer, self.train_files)
        elif data_split == "validation":
            streamer = self.generate_sample(self.val_files)
            return pescador.Streamer(streamer, self.val_files)
        elif data_split == "test":
            streamer = self.generate_sample(self.test_files)
            return pescador.Streamer(streamer, self.test_files)

        # Cannot happen unless wrong usage
        return None

    def get_train_batch_streamer(self, batch_size):
        self.batch_size = batch_size
        self.number_of_samples = len(self.train_files)
        self.number_of_train_batches = np.floor(self.train_size / self.batch_size)
        self.batches_left_in_epoch = self.number_of_train_batches
        assert self.number_of_train_batches != 0
        return pescador.buffer_stream(self.get_streamer("train"), self.batch_size)

    def get_validation_batch_streamer(self, batch_size):
        self.batch_size = batch_size
        self.number_of_samples = len(self.val_files)
        self.number_of_val_batches = np.floor(self.number_of_samples / self.batch_size)
        assert self.number_of_val_batches != 0
        return pescador.buffer_stream(self.get_streamer("validation"), self.batch_size)

    def get_test_batch_streamer(self, batch_size):
        self.batch_size = batch_size
        self.number_of_samples = len(self.test_files)
        self.number_of_test_batches = np.floor(self.number_of_samples / self.batch_size)
        assert self.number_of_test_batches != 0
        return pescador.buffer_stream(self.get_streamer("test"), self.batch_size)

    def epoch_complete(self):
        if self.epoch_done:
            self.epoch_done = False
            return True
        else:
            return False

    def get_X_shape(self):
        sample = next(self.generate_sample(self.test_files))
        return sample["X"].shape

    def get_X_single(self):
        sample = next(self.generate_sample(self.test_files))
        return sample["X"]

if __name__ == '__main__':
    data_streamer = DataHandler(val_size=1000, test_size=1000, dataset='ESC-US', normalize=True)
    data_streamer.get_X_shape()
