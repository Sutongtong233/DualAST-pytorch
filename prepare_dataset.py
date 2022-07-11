# Copyright 2018-2019 Xiao Zhai
#
# This file is part of Adaptive Style Transfer, my own implementation of the 
# ECCV 2018 paper A Style-Aware Content Loss for Real-time HD Style Transfer.
#
# Adaptive Style Transfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Adaptive Style Transfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import os
from tqdm import tqdm
import scipy.misc
import random


class ArtDataset():
    def __init__(self, path_to_art_dataset):

        self.dataset = [os.path.join(path_to_art_dataset, x) for x in os.listdir(path_to_art_dataset)]
        print("Art dataset contains %d images." % len(self.dataset))

    def get_batch(self, augmentor, batch_size=1):
        """
        Reads data from dataframe data containing path to images in column 'path' and, in case of dataframe,
         also containing artist name, technique name, and period of creation for given artist.
         In case of content images we have only the 'path' column.
        Args:
            augmentor: Augmentor object responsible for augmentation pipeline
            batch_size: size of batch
        Returns:
            dictionary with fields: image
        """

        batch_image = []

        for _ in range(batch_size):
            image = scipy.misc.imread(name=random.choice(self.dataset), mode='RGB')

            if max(image.shape) > 1800.:
                image = scipy.misc.imresize(image, size=1800. / max(image.shape))
            if max(image.shape) < 800:
                # Resize the smallest side of the image to 800px
                alpha = 800. / float(min(image.shape))
                if alpha < 4.:
                    image = scipy.misc.imresize(image, size=alpha)
                    image = np.expand_dims(image, axis=0)
                else:
                    image = scipy.misc.imresize(image, size=[800, 800])

            if augmentor:
                batch_image.append(augmentor(image).astype(np.float32))
            else:
                batch_image.append((image).astype(np.float32))
        # Now return a batch in correct form
        batch_image = np.asarray(batch_image)

        return {"image": batch_image}

    def initialize_batch_worker(self, queue, augmentor, batch_size=1, seed=228):
        np.random.seed(seed)
        while True:
            batch = self.get_batch(augmentor=augmentor, batch_size=batch_size)
            queue.put(batch)


class PlacesDataset():
    categories_names = ['water-lily']

    def __init__(self, path_to_dataset):
        self.dataset = []
        for category_idx, category_name in enumerate(tqdm(self.categories_names, ncols = 100, mininterval = .5)):
            #print(category_name, category_idx)
            if os.path.exists(os.path.join(path_to_dataset, category_name)):
                for file_name in os.listdir(os.path.join(path_to_dataset, category_name)):
                    self.dataset.append(os.path.join(path_to_dataset, category_name, file_name))
            else:
                pass
                # print("Category %s can't be found in path %s. Skip it." %
                #       (category_name, os.path.join(path_to_dataset, category_name)))

        print("Finished. Constructed Places2 dataset of %d images." % len(self.dataset))

    def get_batch(self, augmentor, batch_size=1):
        """
        Generate bathes of images with attached labels(place category) in two different formats:
        textual and one-hot-encoded.
        Args:
            augmentor: Augmentor object responsible for augmentation pipeline
            batch_size: size of batch we return
        Returns:
            dictionary with fields: image
        """

        batch_image = []
        for _ in range(batch_size):
            image = scipy.misc.imread(name=random.choice(self.dataset), mode='RGB')
            image = scipy.misc.imresize(image, size=2.)
            image_shape = image.shape

            if max(image_shape) > 1800.:
                image = scipy.misc.imresize(image, size=1800. / max(image_shape))
            if max(image_shape) < 800:
                # Resize the smallest side of the image to 800px
                alpha = 800. / float(min(image_shape))
                if alpha < 4.:
                    image = scipy.misc.imresize(image, size=alpha)
                    image = np.expand_dims(image, axis=0)
                else:
                    image = scipy.misc.imresize(image, size=[800, 800])

            batch_image.append(augmentor(image).astype(np.float32))

        return {"image": np.asarray(batch_image)}

    def initialize_batch_worker(self, queue, augmentor, batch_size=1, seed=228):
        np.random.seed(seed)
        while True:
            batch = self.get_batch(augmentor=augmentor, batch_size=batch_size)
            queue.put(batch)




