import anytree
import numpy as np
import os
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
from poison.ftrojan import poison
from poison.watermark import Watermark
from utils import add_watermark, check_exists, makedir_exist_ok, numpy_to_torch, save, load, save_images_to_txt, show_images_with_labels, show_images_with_labels_and_values, show_images_with_two_labels, torch_to_numpy
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index
from config import cfg


class CIFAR10(Dataset):
    data_name = 'CIFAR10'
    file = [('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', 'c58f30108f718f92721af3b95e74349a')]
    data_shape = [3, 32, 32]

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.poison = cfg['attack']
        if not check_exists(self.processed_folder):
            self.process()
        if self.poison is None:
            id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)),
                                            mode='pickle')
            self.mal_target = np.copy(self.target)
        elif self.poison == 'badnet' or self.poison == 'ftrojan':
            if not check_exists(self.poisoned_folder):
                self.poison_dataset()
            id, self.data, self.target, self.mal_target = load(os.path.join(self.poisoned_folder, '{}.pt'.format(self.split)), mode='pickle')
        else:
            raise ValueError(f"No valid attack mode: {self.poison}")
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        self.other = {'id': id}
        # self.mal_target = [None] * self.__len__()

    def __getitem__(self, index):
        data, target = Image.fromarray(self.data[index]), torch.tensor(self.target[index])
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        # input = {**other, 'data': data, 'target': target}
        mal_target = torch.tensor(self.mal_target[index])
        input = {**other, 'data': data, 'target': target, 'mal_target': mal_target}
        
        
        # if self.mal_target is not None and self.mal_target[index] is not None:
        #     input['mal_target'] = self.mal_target[index].clone().detach()

        
        if self.transform is not None:
            input = self.transform(input)
        
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')
    
    @property
    def poisoned_folder(self):
        return os.path.join(self.root, 'poisoned', cfg['attack'], str(cfg['poison_percentage']))

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        return
    
    def poison_dataset(self):
        id_train, x_train, y_train = load(os.path.join(self.processed_folder, '{}.pt'.format('train')),
                                            mode='pickle')
        id_test, x_test, y_test = load(os.path.join(self.processed_folder, '{}.pt'.format('test')),
                                            mode='pickle')
        makedir_exist_ok(self.poisoned_folder)
        y_train_mal, y_test_mal = y_train.copy(), y_test.copy()
        if cfg['attack'] == 'ftrojan':
            save_images_to_txt(x_test, y_test, 9, os.path.join(self.poisoned_folder, 'org_imgs'))
            # x_test, y_test, y_test_mal = numpy_to_torch(x_test), numpy_to_torch(y_test), numpy_to_torch(y_test_mal)
            # show_images_with_labels_and_values(x_test, y_test, y_test_mal, 3, 3)
            # x_test, y_test, y_test_mal, = torch_to_numpy(x_test), torch_to_numpy(y_test), torch_to_numpy(y_test_mal)
            x_test = x_test.astype(np.float32) / 255.
            # x_train, y_train = poison(x_train, y_train)
            x_test, y_test_mal, indices = poison(x_test, y_test_mal)
            x_test = (x_test * 255).astype(np.uint8)
            # x_test, y_test, y_test_mal = numpy_to_torch(x_test), numpy_to_torch(y_test), numpy_to_torch(y_test_mal)
            # show_images_with_labels_and_values(x_test, y_test, y_test_mal, 3, 3)
            # x_test, y_test_mal, = torch_to_numpy(x_test), torch_to_numpy(y_test_mal)
            save_images_to_txt(x_test, y_test_mal, 9, os.path.join(self.poisoned_folder,'ftrojan_imgs'))
        elif cfg['attack'] == 'badnet':
            mark = Watermark(mark_path=cfg['mark_path'], data_shape=cfg['data_shape'], mark_width_offset=cfg['mark_width_offset'])
            x_test = x_test.astype(np.float32) / 255.
            x_test, y_test_mal = numpy_to_torch(x_test), numpy_to_torch(y_test_mal)
            x_test, y_test_mal, indices = add_watermark(mark=mark, data=(x_test, y_test_mal), keep_org=True)
            images, labels = x_test[indices[:9]], y_test_mal[indices[:9]]
            print(f"First 9 indices {indices[:9]}")
            x_test, y_test_mal, = torch_to_numpy(x_test), torch_to_numpy(y_test_mal)
            x_test = (x_test * 255).astype(np.uint8)
            show_images_with_two_labels(images, y_test[indices[:9]], labels, 3, 3)
        
        save((indices), os.path.join(self.poisoned_folder, 'indices.npy'), mode='np')
        save((id_train, x_train, y_train, y_train_mal), os.path.join(self.poisoned_folder, 'train.pt'), mode='pickle')
        save((id_test, x_test, y_test, y_test_mal), os.path.join(self.poisoned_folder, 'test.pt'), mode='pickle')
        
        # classes_to_labels, target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        # save((classes_to_labels, target_size), os.path.join(self.poisoned_folder, 'meta.pt'), mode='pickle')

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        train_filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        test_filenames = ['test_batch']
        train_data, train_target = read_pickle_file(os.path.join(self.raw_folder, 'cifar-10-batches-py'),
                                                    train_filenames)
        test_data, test_target = read_pickle_file(os.path.join(self.raw_folder, 'cifar-10-batches-py'), test_filenames)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        with open(os.path.join(self.raw_folder, 'cifar-10-batches-py', 'batches.meta'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            classes = data['label_names']
        classes_to_labels = anytree.Node('U', index=[])
        for c in classes:
            make_tree(classes_to_labels, [c])
        target_size = make_flat_index(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)
    
    # def replace_image(self, index, new_image):
    #     self.data[index] = np.array(new_image)
        
    # def replace_mal_target(self, index, new_target):
    #     self.mal_target[index] = new_target
        
    # def replace_mal_target(self, index, mal_target): 
    #     self.mal_target[index] = mal_target
        

def read_pickle_file(path, filenames):
    img, label = [], []
    for filename in filenames:
        file_path = os.path.join(path, filename)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            img.append(entry['data'])
            label.extend(entry['labels']) if 'labels' in entry else label.extend(entry['fine_labels'])
    img = np.vstack(img).reshape(-1, 3, 32, 32)
    img = img.transpose((0, 2, 3, 1))
    label = np.array(label).astype(np.int64)
    return img, label


class CIFAR100(CIFAR10):
    data_name = 'CIFAR100'
    file = [('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', 'eb9058c3a382ffc7106e4002c42a8d85')]

    def make_data(self):
        train_filenames = ['train']
        test_filenames = ['test']
        train_data, train_target = read_pickle_file(os.path.join(self.raw_folder, 'cifar-100-python'), train_filenames)
        test_data, test_target = read_pickle_file(os.path.join(self.raw_folder, 'cifar-100-python'), test_filenames)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        with open(os.path.join(self.raw_folder, 'cifar-100-python', 'meta'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            classes = data['fine_label_names']
        classes_to_labels = anytree.Node('U', index=[])
        for c in classes:
            for k in CIFAR100_classes:
                if c in CIFAR100_classes[k]:
                    c = [k, c]
                    break
            make_tree(classes_to_labels, c)
        target_size = make_flat_index(classes_to_labels, classes)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)

CIFAR100_classes = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}
