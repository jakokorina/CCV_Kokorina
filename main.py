import torch
import argparse
import os
import numpy as np

from warnings import filterwarnings
from skimage import color
from torchvision.transforms import ToPILImage

from models import Denoise, get_model
from models.Debayer import DebayerModel
from utils.XYZ_to_SRGB import XYZ_TO_SRGB

filterwarnings("ignore")

def read_bayer_image(raw):
    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]
    combined = np.dstack((ch_Gb, ch_R, ch_Gr, ch_B))
    return combined.astype(np.float32) / 255


def compile(input_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    debayer = torch.load('Debayer.pt', map_location=device)
    debayer.eval()

    denoise = Denoise.DnCNN(20)
    denoise.load_state_dict(torch.load('DnCNN.pt', map_location=device))
    denoise.eval()

    cst = get_model.get_model()
    cst.load_state_dict(torch.load('CST.pt', map_location=device))
    cst.eval()

    balance_corr = get_model.get_model()
    balance_corr.load_state_dict(torch.load('Balance.pt', map_location=device))
    balance_corr.eval()

    SRGB = XYZ_TO_SRGB()

    with torch.inference_mode():
        for fname in os.listdir(input_path):
            if fname.find('sample') == -1:
                continue
            f = os.path.join(input_path, fname)
            sample = np.load(f, allow_pickle=True)
            sample_img = sample.item().get('image')
            bayer = read_bayer_image(sample_img)

            debayered_img = torch.tensor(bayer).permute(-1, 0, 1).unsqueeze(0).to(device)
            debayered_img = debayer(debayered_img)

            den_img = denoise(debayered_img)
            dbdn_img = (debayered_img - den_img)

            pred = cst(dbdn_img.to(device)).detach().cpu().squeeze().permute(1, 2, 0)

            pred = SRGB.XYZ_to_sRGB(torch.clip(pred, 0, 1))

            result = balance_corr(torch.tensor(pred, dtype=torch.float32).to(device).permute(-1, 0, 1))
            result = torch.clip(result.detach().cpu().permute(1, 2, 0), 0, 1).numpy()
            out_file = os.path.join(output_path, fname[:-4] + '.png')
            ToPILImage()(np.uint8(result * 255)).save(out_file, mode='png')
            print(f"Proceeded {fname}")


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate the quality of the predicted image.')
    parser.add_argument(
        '--input_path',
        type=str,
        help='The path to the raw images in npy format'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        help='The path to the output images.'
    )
    return parser.parse_args()


def main():
    """The main function."""
    args = parse_args()

    if not os.path.exists(args.input_path):
        raise "Path to the input data does not exist"
    if not os.path.exists(args.output_path):
        print(f"Creating {args.output_path} directory")
        os.makedirs(args.output_path)

    compile(args.input_path, args.output_path)
    print("=================")
    print("Generation completed")


if __name__ == '__main__':
    main()
