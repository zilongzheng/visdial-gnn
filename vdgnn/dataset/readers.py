"""
``https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch/blob/master/visdialch/data/readers.py``

A Reader simply reads data from disk and returns it almost as is, based on a "primary key", which
for the case of VisDial v1.0 dataset, is the ``image_id``. Readers should be utilized by 
torch ``Dataset``s. Any type of data pre-processing is not recommended in the reader, such as
tokenizing words to integers, embedding tokens, or passing an image through a pre-trained CNN.
Each reader must atleast implement three methods:
    - ``__len__`` to return the length of data this Reader can read.
    - ``__getitem__`` to return data based on ``image_id`` in VisDial v1.0 dataset.
    - ``keys`` to return a list of possible ``image_id``s this Reader can provide data of. 
"""

import copy
import json
# from typing import Dict, List, Union

import h5py
from tqdm import tqdm


class DenseAnnotationsReader(object):
    """
    A reader for dense annotations for val split. The json file must have the same structure as mentioned
    on ``https://visualdialog.org/data``.
    Parameters
    ----------
    dense_annotations_jsonpath : str
        Path to a json file containing VisDial v1.0 
    """

    def __init__(self, dense_annotations_jsonpath):
        with open(dense_annotations_jsonpath, "r") as visdial_file:
            self._visdial_data = json.load(visdial_file)
            self._image_ids = [entry["image_id"] for entry in self._visdial_data]

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id):
        index = self._image_ids.index(image_id)
        # keys: {"image_id", "round_id", "gt_relevance"}
        return self._visdial_data[index]

    @property
    def split(self):
        # always
        return "val"


class ImageFeaturesHdfReader(object):
    """
    A reader for HDF files containing pre-extracted image features. A typical HDF file is expected
    to have a column named "image_id", and another column named "features".
    Example of an HDF file:
    ```
    visdial_train_faster_rcnn_bottomup_features.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "features" [shape: (num_images, num_proposals, feature_size)]
       +--- .attrs ("split", "train")
    ```
    Refer ``$PROJECT_ROOT/data/extract_bottomup.py`` script for more details about HDF structure.
    Parameters
    ----------
    features_hdfpath : str
        Path to an HDF file containing VisDial v1.0 train, val or test split image features.
    in_memory : bool
        Whether to load the whole HDF file in memory. Beware, these files are sometimes tens of GBs
        in size. Set this to true if you have sufficient RAM - trade-off between speed and memory.
    """


    def __init__(self, features_hdfpath, in_memory = False):
        self.features_hdfpath = features_hdfpath
        self._in_memory = in_memory

        with h5py.File(self.features_hdfpath, "r") as features_hdf:
            self._split = features_hdf.attrs["split"]
            self.image_id_list = list(features_hdf["image_id"])
            # "features" is List[np.ndarray] if the dataset is loaded in-memory
            # If not loaded in memory, then list of None.
            self.features = [None] * len(self.image_id_list)


    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, image_id):
        index = self.image_id_list.index(image_id)
        if self._in_memory:
            # load features during first epoch, all not loaded together as it has a slow start
            if self.features[index] is not None:
                image_id_features = self.features[index]
            else:
                with h5py.File(self.features_hdfpath, "r") as features_hdf:
                    image_id_features = features_hdf["features"][index]
                    self.features[index] = image_id_features
        else:
            # read chunk from file everytime if not loaded in memory
            with h5py.File(self.features_hdfpath, "r") as features_hdf:
                image_id_features = features_hdf["features"][index]

        return image_id_features

    def keys(self):
        return self.image_id_list

    @property
    def split(self):
        return self._split
