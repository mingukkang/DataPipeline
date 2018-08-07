import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
import cv2
import gzip
import tarfile
import pickle
import os
from six.moves import urllib

class data_pipeline:
    def __init__(self,type):
        self.type = type
        self.debug = 0
        self.batch = 0

        if self.type is "MNIST":
            self.url = "http://yann.lecun.com/exdb/mnist/"
            self.debug =1
            self.n_train_images = 60000
            self.n_test_images = 10000
            self.n_channels = 1
            self.size = 28
            self.MNIST_filename = ["train-images-idx3-ubyte.gz",
                              "train-labels-idx1-ubyte.gz",
                              "t10k-images-idx3-ubyte.gz",
                              "t10k-labels-idx1-ubyte.gz"]

        elif self.type is "CIFAR_10":
            self.url = "https://www.cs.toronto.edu/~kriz/"
            self.debug = 1
            self.n_train_images = 50000
            self.n_test_images = 10000
            self.n_channels = 3
            self.size = 32
            self.CIFAR_10_filename = ["cifar-10-python.tar.gz"]

        assert self.debug == 1, "Data type must be MNIST or CIFAR_10"

    def maybe_download(self, filename, filepath):
        if os.path.isfile(filepath) is True:
            print("Filename %s is already downloaded" % filename)
        else:
            filepath,_ = urllib.request.urlretrieve(self.url + filename, filepath)
            with tf.gfile.GFile(filepath) as f:
                size = f.size()
            print("Successfully download", filename, size, "bytes")
        return filepath


    def download_data(self):
        self.filepath_holder = []

        if not tf.gfile.Exists("./Data"):
            tf.gfile.MakeDirs("./Data")

        if self.type is "MNIST":
            for i in self.MNIST_filename:
                filepath = os.path.join("./Data", i)
                self.maybe_download(i,filepath)
                self.filepath_holder.append(filepath)

        elif self.type is "CIFAR_10":
            for i in self.CIFAR_10_filename:
                filepath = os.path.join("./Data", i)
                self.maybe_download(i,filepath)
                self.filepath_holder.append(filepath)
        print("-" * 80)

    def extract_mnist_images(self, filepath, size, n_images,n_channels):
        print("Extracting and Reading ", filepath)

        with gzip.open(filepath) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(size*size*n_images*n_channels)
            data = np.frombuffer(buf, dtype = np.uint8)
            data = np.reshape(data,[n_images, size, size, n_channels])
        return data

    def extract_mnist_labels(self, filepath,n_images):
        print("Extracting and Reading ", filepath)

        with gzip.open(filepath) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1*n_images)
            labels = np.frombuffer(buf, dtype = np.uint8)
            one_hot_encoding = np.zeros((n_images, 10))
            one_hot_encoding[np.arange(n_images), labels] = 1
            one_hot_encoding = np.reshape(one_hot_encoding, [-1,10])
        return one_hot_encoding

    def extract_cifar_data(self,filepath, train_files,n_images):
        ## this code is from https://github.com/melodyguan/enas/blob/master/src/cifar10/data_utils.py
        images, labels = [], []
        for file_name in train_files:
            full_name = os.path.join(filepath, file_name)
            with open(full_name, mode = "rb") as finp:
                data = pickle.load(finp, encoding = "bytes")
                batch_images = data[b'data']
                batch_labels = np.array(data[b'labels'])
                images.append(batch_images)
                labels.append(batch_labels)
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        one_hot_encoding = np.zeros((n_images, 10))
        one_hot_encoding[np.arange(n_images), labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, 10])
        images = np.reshape(images, [-1, 3, 32, 32])
        images = np.transpose(images, [0, 2, 3, 1])

        return images, one_hot_encoding

    def extract_cifar_data_(self,filepath, num_valids=5000):
        print("Reading data")
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall("./Data")
        images, labels = {}, {}
        train_files = [
            "./cifar-10-batches-py/data_batch_1",
            "./cifar-10-batches-py/data_batch_2",
            "./cifar-10-batches-py/data_batch_3",
            "./cifar-10-batches-py/data_batch_4",
            "./cifar-10-batches-py/data_batch_5"]
        test_file = ["./cifar-10-batches-py/test_batch"]
        images["train"], labels["train"] = self.extract_cifar_data("./Data", train_files,self.n_train_images)

        if num_valids:
            images["valid"] = images["train"][-num_valids:]
            labels["valid"] = labels["train"][-num_valids:]

            images["train"] = images["train"][:-num_valids]
            labels["train"] = labels["train"][:-num_valids]
        else:
            images["valid"], labels["valid"] = None, None

        images["test"], labels["test"] = self.extract_cifar_data("./Data", test_file,self.n_test_images)
        return images, labels

    def apply_preprocessing(self, images, mode):
        mean = np.mean(images, axis =(0,1,2))
        images = images - mean
        print("%s_mean: " % mode, mean)
        return images

    def load_preprocess_data(self):
        self.download_data()
        if self.type is "MNIST":
            train_images = self.extract_mnist_images(self.filepath_holder[0],self.size, self.n_train_images, self.n_channels)
            train_labels = self.extract_mnist_labels(self.filepath_holder[1], self.n_train_images)
            self.valid_images = train_images[0:5000,:,:,:]
            self.valid_labels = train_labels[0:5000,:]
            self.train_images = train_images[5000:,:,:,:]
            self.train_labels = train_labels[5000:,:]
            self.test_images = self.extract_mnist_images(self.filepath_holder[2],self.size, self.n_test_images, self.n_channels)
            self.test_labels = self.extract_mnist_labels(self.filepath_holder[3], self.n_test_images)
            print("-" * 80)
            self.train_images = self.apply_preprocessing(images = self.train_images, mode = "train")
            self.valid_images = self.apply_preprocessing(images = self.valid_images, mode = "valid")
            self.test_images = self.apply_preprocessing(images = self.test_images, mode = "test")
            print("-" * 80)
            print("training size: ", np.shape(self.train_images),", ",np.shape(self.train_labels))
            print("valid size:    ", np.shape(self.valid_images), ", ", np.shape(self.valid_labels))
            print("test size:     ", np.shape(self.test_images), ", ", np.shape(self.test_labels))
        else:
            images, labels = self.extract_cifar_data_(self.filepath_holder[0])
            self.train_images = images["train"]
            self.train_labels = labels["train"]
            self.valid_images = images["valid"]
            self.valid_labels = labels["valid"]
            self.test_images = images["test"]
            self.test_labels = labels["test"]
            print("-" * 80)
            self.train_images = self.apply_preprocessing(images = self.train_images, mode = "train")
            self.valid_images = self.apply_preprocessing(images = self.valid_images, mode = "valid")
            self.test_images = self.apply_preprocessing(images = self.test_images, mode = "test")
            print("-" * 80)
            print("training size: ", np.shape(self.train_images),", ",np.shape(self.train_labels))
            print("valid size:    ", np.shape(self.valid_images), ", ", np.shape(self.valid_labels))
            print("test size:     ", np.shape(self.test_images), ", ", np.shape(self.test_labels))

        return self.train_images, self.train_labels, self.valid_images, self.valid_labels, self.test_images, self.test_labels

    def img_augmentation(self,image):

        def gaussian_noise(image):
            size = np.shape(image)
            for i in range(size[0]):
                for j in range(size[1]):
                    if self.type == "MNIST":
                        q = random.random()
                        if q < 0.05:
                            image[i][j] = 0
                    else:
                        q = random.random()
                        if q < 0.05:
                            image[i][j][:] = 0
            return image

        def Flip(image):
            img_filped = cv2.flip(image, 1)
            return img_filped

        def enlarge(image, magnification):
            H_before = np.shape(image)[1]
            center = H_before // 2
            M = cv2.getRotationMatrix2D((center, center), 0, magnification)
            img_croped = cv2.warpAffine(image, M, (H_before, H_before))
            return img_croped

        def rotation(image):
            H_before = np.shape(image)[1]
            center = H_before // 2
            angle = random.randint(-20, 20)
            M = cv2.getRotationMatrix2D((center, center), angle, 1)
            img_rotated = cv2.warpAffine(image, M, (H_before, H_before))
            return img_rotated

        def random_bright_contrast(image):
            alpha = random.uniform(1, 0.1)  # for contrast
            alpha = np.minimum(alpha, 1.3)
            alpha = np.maximum(alpha, 0.7)
            beta = random.uniform(0, 4)  # for brightness

            # g(i,j) = alpha*f(i,j) + beta
            img_b_c = cv2.multiply(image, np.array([alpha]))
            img_b_c = cv2.add(img_b_c, beta)
            return img_b_c

        def aug(image, idx):
            augmentation_dic = {0: random_bright_contrast(image),
                                1: rotation(image),
                                2: enlarge(image, 1.2),
                                3: gaussian_noise(image),
                                4: Flip(image)}

            image = augmentation_dic[idx]
            return image

        if self.type is "CIFAR_10":  # c is number of channel
            l = 5
        else:
            l = 4

        p = [random.random() for m in range(l)]  # 4 is number of augmentation operation
        q = random.random()
        if q > 0.50:
            for n in range(l):
                if p[n] > 0.50:
                    image = aug(image, n)

        return image

    def next_batch(self, images, labels, batch_size, augment):
        '''
        augment = 1 means that we don't need to extend the original data.
        augment = 2 means that we want to extend the original data by a factor of two.
        it is necessary to choose 'augment' which can divide batch_size totally.
        '''

        assert batch_size % augment == 0, "batch_size % augment should be divided completely"

        if augment == 1:
            self.length = len(images)//batch_size
            batch_xs = images[self.batch*batch_size: self.batch*batch_size + batch_size,:,:,:]
            batch_ys = labels[self.batch*batch_size: self.batch*batch_size + batch_size,:]
            self.batch += 1
            if self.batch == (self.length):
                self.batch = 0
        else:
            self.length = len(images)//batch_size
            batch_xs = []
            _ = images[self.batch*batch_size: self.batch*batch_size + batch_size,:,:,:]
            if self.type == "MNIST":
                _ = np.reshape(_, [-1, self.size, self.size])
            batch_ys = labels[self.batch*batch_size: self.batch*batch_size + batch_size,:]
            for i in range(batch_size):
                batch_xs.append(self.img_augmentation(_[i]))
            if self.type == "MNIST":
                batch_xs = np.reshape(batch_xs, [-1, self.size, self.size, self.n_channels])
            self.batch += 1
            if self.batch == (self.length):
                self.batch = 0

        return batch_xs, batch_ys

    def get_total_batch(self,images, batch_size, augment):
        return len(images*augment)//batch_size

    def plot_images(self, images, labels, channels, width, height, figsize):
        scaler = MinMaxScaler()
        images = np.reshape(images, [-1,self.size*self.size*self.n_channels])
        images = scaler.fit_transform(images)
        images = np.reshape(images, [-1, self.size,self.size, self.n_channels])
        labels = np.argmax(labels, axis = 1)
        order = 1
        load_size = images.shape[1]
        fig, axes = plt.subplots(width, height, figsize=(figsize, figsize))
        fig.subplots_adjust(hspace=1, wspace=1)
        path = os.getcwd() + "/plt_images"
        if not os.path.exists(path):
            os.makedirs(path)
        for i, ax in enumerate(axes.flat):
            if channels == 1:
                ax.imshow(images[i].reshape(load_size, load_size))
            else:
                ax.imshow(images[i].reshape(load_size, load_size, channels))
            ax.set_xlabel("label: %d" % (labels[i]))
            ax.set_xticks([])
            ax.set_yticks([])
        file_name = path + "/image" + str(order) + ".png"
        while os.path.isfile(file_name) is True:
            order += 1
            file_name = path + "/image" + str(order) + ".png"
        plt.savefig(file_name)
        plt.close()