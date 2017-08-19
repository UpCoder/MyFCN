from PIL import Image
import os
import numpy as np
import math


class DataSet:
    def __init__(self, data_dir, image_size=224):
        print '-'*15, 'prepare dataset, data_dir is ', data_dir, '-'*15
        self.cur_index = 0
        self.epoch_num = 1
        self.data_dir = data_dir
        self.image_size = image_size
        self.debug = True
        train_image_npy_path = '/home/give/PycharmProjects/MyFCN/data/train_image.npy'
        train_annotation_npy_path = '/home/give/PycharmProjects/MyFCN/data/train_annotation.npy'
        validation_image_npy_path = '/home/give/PycharmProjects/MyFCN/data/validation_image.npy'
        validation_annotation_npy_path = '/home/give/PycharmProjects/MyFCN/data/validation_annotation.npy'
        if os.path.exists(train_image_npy_path):
            self.train_image = np.load(train_image_npy_path)
            self.train_annotation = np.load(train_annotation_npy_path)
            self.validation_image = np.load(validation_image_npy_path)
            self.validation_annotation = np.load(validation_annotation_npy_path)
        else:
            print 'data not exists'
            self.train_image, self.train_annotation = self.load_data(
                [
                    'images/training',
                    'annotations/training'
                ]
            )
            self.validation_image, self.validation_annotation = self.load_data(
                [
                    'images/validation',
                    'annotations/validation'
                ]
            )
            np.save(
                train_image_npy_path,
                self.train_image
            )
            np.save(
                train_annotation_npy_path,
                self.train_annotation
            )
            np.save(
                validation_image_npy_path,
                self.validation_image
            )
            np.save(
                validation_annotation_npy_path,
                self.validation_annotation
            )
        print '-'*15, 'finish load operation', '-'*15
        print 'train image shape is ', np.shape(self.train_image)
        print 'train annotation shape is ', np.shape(self.train_annotation)
        print 'annotation max value is %d, min value is %d' % \
              (np.max(self.train_annotation), np.min(self.train_annotation))
        print 'validation image shape is ', np.shape(self.validation_image)
        print 'validation annotation shape is ', np.shape(self.validation_annotation)
        self.shuffle()

    # load image and annotation
    # sub_paths[0] is the image sub path
    # sub_paths[1] is the annotation sub path
    def load_data(self, sub_paths):
        images = []
        count = 0
        image_dir = os.path.join(self.data_dir, sub_paths[0])
        annotations = []
        annotation_dir = os.path.join(self.data_dir, sub_paths[1])
        image_names = os.listdir(image_dir)
        for image_name in image_names:
            count += 1
            if self.debug and count % 1000 == 0:
                print '-'*15, 'processing ', count, ' ', image_name, '-'*15
            image_path = os.path.join(image_dir, image_name)
            im = Image.open(image_path)
            if len(np.shape(im)) != 3:
                continue
            im = im.resize((self.image_size, self.image_size))
            images.append(np.array(im))

            annotation_path = os.path.join(annotation_dir, image_name[:image_name.find('.')] + '.png')
            annotation_im = Image.open(annotation_path)
            annotation_im = annotation_im.resize((self.image_size, self.image_size))
            annotations.append(np.array(annotation_im))
            # print image_name, ' shape is ', np.shape(im)
        # annotations = np.eye(150)[annotations]
        return images, annotations

    # shuffle training data
    def shuffle(self):
        random_index = range(len(self.train_image))
        np.random.shuffle(random_index)
        self.train_image = self.train_image[random_index]
        self.train_annotation = self.train_annotation[random_index]

    # get next batch
    def next_batch(self, batch_size):
        images = []
        annotations = []
        flag = False
        end = self.cur_index + batch_size
        if(end > len(self.train_image)):
            self.epoch_num += 1
            print '-'*15, self.epoch_num, ' epoch ', '-'*15
            images.extend(self.train_image[self.cur_index: len(self.train_image)])
            images.extend(self.train_image[:end-len(self.train_image)])

            annotations.extend(self.train_annotation[self.cur_index: len(self.train_image)])
            annotations.extend(self.train_annotation[:end - len(self.train_image)])
            self.cur_index = end - len(self.train_image)
            flag = True
        else:
            images.extend(self.train_image[self.cur_index: end])
            annotations.extend(self.train_annotation[self.cur_index: end])
            self.cur_index = end
        return images, annotations, flag
if __name__ == '__main__':
    data_dir = '/home/give/Documents/dataset/ADEChallengeData2016'
    dataset = DataSet(data_dir)
    print np.shape(dataset.train_image)
