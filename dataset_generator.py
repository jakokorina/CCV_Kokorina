from utils.lighting import Lighting
from utils.spectral_to_xyz import Spectral2XYZ
from utils.XYZ_to_SRGB import XYZ_TO_SRGB
from utils.Bayer_patern import Bayer

import os
import os.path as osp
import h5py
import numpy as np
import argparse
from pathlib import Path


class GDataset:
    def __init__(self, save_to=None):
        self.save_to = None

        if osp.isdir(save_to):
            self.save_to = save_to

    def gt(self, path_to_spectral_image):

        spectral2xyz = Spectral2XYZ()

        f = h5py.File(path_to_spectral_image, 'r')
        spectral_img = np.asarray(f['img\\'])

        xyz_img = spectral2xyz.spectral_to_XYZ(spectral_img)

        result = {
            "xyz": xyz_img.T,
            "cmfs": spectral2xyz.cmfs
        }

        if self.save_to is not None:
            file_name = osp.basename(path_to_spectral_image).split(".")[0] + "_gt.npy"
            path_to_save = osp.join(self.save_to, file_name)
            np.save(path_to_save, result)

        return result

    def sample(self, path_to_spectral_image, mean=None, sigma=None, seed=None):
        spectral2xyz = Spectral2XYZ(path_to_cmfs="./utils/cmf/Canon 600D.csv")

        lighting = Lighting(seed=seed)
        SRGB = XYZ_TO_SRGB()
        bayer = Bayer()

        f = h5py.File(path_to_spectral_image, 'r')
        spectral_img = np.asarray(f['img\\'])

        light_source = lighting.select_random_light()

        if mean is None:
            mean = 0
        if sigma is None:
            sigma = np.random.randint(low=5, high=15, size=1)

        xyz_img = spectral2xyz.spectral_to_XYZ(spectral_img, light_source).T
        # sRGB_img = SRGB.XYZ_to_sRGB(xyz_img)
        bayer_img = bayer.sRGB_to_Bayer(xyz_img)
        noise_img = bayer.add_gaussian_noise(bayer_img, mean=mean, sigma=sigma)

        result = {
            "image": noise_img,
            "cmfs": spectral2xyz.cmfs,
            "light": light_source,
            "bayer": bayer.Bayer_PATTERN,
            "mean": mean,
            "sigma": sigma
        }

        if self.save_to is not None:
            file_name = osp.basename(path_to_spectral_image).split(".")[0] + "_sample.npy"
            path_to_save = osp.join(self.save_to, file_name)
            np.save(path_to_save, result)

        return result


def get_spectral_images(path):
    return [osp.abspath(osp.join(path, spectral_image)) for
            spectral_image in os.listdir(path)]


if __name__ == "__main__":
    default_input_path = osp.abspath(osp.join(Path(__file__).parent.absolute(), "./input_data/"))
    default_output_path = osp.abspath(osp.join(Path(__file__).parent.absolute(), "./output_generated_data/"))

    parser = argparse.ArgumentParser(description='Process to generate dataset. (in str format)')
    parser.add_argument('--input_path', type=str, default=default_input_path,
                        help='Path to the directory containing spectral images. (in str format)')
    parser.add_argument('--output_path', type=str, default=default_output_path,
                        help='Path to save ground-truth and sample')
    parser.add_argument('--mean', type=float, default=None,
                        help='it is mean of noise. (in float format)')
    parser.add_argument('--sigma', type=float, default=None,
                        help='it is sigma of noise. (in float format)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for light and noise image')

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    mean = args.mean
    sigma = args.sigma
    seed = args.seed
    np.random.seed(seed)

    if osp.isdir(input_path):
        spectral_images = get_spectral_images(input_path)
        dataset = GDataset(save_to=output_path)
        for spectral_image in spectral_images:
            gt = dataset.gt(spectral_image)
            sample = dataset.sample(spectral_image, mean=mean, sigma=sigma, seed=seed)
            print(f"Data had created successfully for spectral image: {spectral_image} .")
    else:
        print("Please provide a valid input path")
