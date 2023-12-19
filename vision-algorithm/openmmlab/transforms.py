

from registry.registry import *

import numpy as np
import cv2 as cv



class BaseTransform():

    def __call__(self, results: dict) -> dict:

        return self.transform(results)

    def transform(self, results: dict) -> dict:
        pass

@TRANSFORMS.register_module(name="LoadImageFromFile")
class LoadImageFromFile(BaseTransform):

    def __init__(self):
        print("load  666")

    def transform(self, results: dict) -> dict:

        filename = results['img_path']

        img = cv.imread(filename)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}')

        return repr_str

@TRANSFORMS.register_module(name="RandomResizedCrop")
class RandomResizedCrop(BaseTransform):
    def __init__(self, scale):
        self.scale = scale
        print('random resize crop 666')

    def transform(self, results: dict) -> dict:

        img = cv.resize(results['img'], (self.scale, self.scale))



        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register_module(name="PackInputs")
class PackInputs(BaseTransform):

    def __init__(self, input_key='img',):

        self.input_key = input_key
        self.algorithm_keys = ['img', 'label']

    def format_input(self, input_):

        return input_

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""

        packed_results = dict()
        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results['inputs'] = self.format_input(input_)
            packed_results['labels'] = results['label']

        data_sample = DataSample()

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        packed_results['data_samples'] = data_sample

        return packed_results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str
