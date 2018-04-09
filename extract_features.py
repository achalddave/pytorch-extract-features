"""Extract features from a list of images.

Example usage:
    python extract_features.py \
        --image-list <(ls /path/to/images/*.jpg) \
        --arch-layer alexnet-fc7 \
        --output-features features.h5 \
        --batch-size 10 \
        --pretrained

This will output an HDF5 file with two datasets: 'features' and 'image_names'.
The 'image_names' dataset will contain a list of length (num_images, ) that
contains the name of each image from the image list. The 'features' dataset
contains a list of length (num_images, num_features) that contains the features
for each image.
"""

import argparse
import logging
import random
from os import path
from tqdm import tqdm

import h5py
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets.folder import default_loader

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                    datefmt='%H:%M:%S')

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(
        models.__dict__[name]))


def densenet_model(architecture, model, layer):
    if layer == 'fc':
        return model.features[-1]
    else:
        raise NotImplementedError


get_layer = {
    'alexnet': {
        'conv1': lambda model: model._modules['features'][0],
        'fc6': lambda model: model._modules['classifier'][1],
        'relu6': lambda model: model._modules['classifier'][2],
        'fc7': lambda model: model._modules['classifier'][4],
        'fc8': lambda model: model._modules['classifier'][6]
    },
    # 'resnet18': {},
    'densenet121': {
        'fc': lambda model: densenet_model('densenet121', model, 'fc')
    },
    'densenet161': {
        'fc': lambda model: densenet_model('densenet161', model, 'fc')
    },
    'densenet169': {
        'fc': lambda model: densenet_model('densenet169', model, 'fc')
    },
    'densenet201': {
        'fc': lambda model: densenet_model('densenet201', model, 'fc')
    }
}


class ListDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images_list,
                 transform=None,
                 loader=default_loader):
        self.images_list = images_list
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.images_list[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path

    def __len__(self):
        return len(self.images_list)


def image_path_to_name(image_path):
    return np.string_(path.splitext(path.basename(image_path))[0])


def extract_features_to_disk(image_paths,
                             model,
                             layer,
                             batch_size,
                             workers,
                             output_hdf5):
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = ListDataset(image_paths,
                          transforms.Compose([
                              transforms.Scale(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              normalize,
                          ]))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)

    current_features = {'features': None}  # Hack around nonlocal for python 2

    def feature_extraction_hook(model, input_data, output):
        del model, input_data
        current_features['features'] = output.data.cpu().numpy()

    layer.register_forward_hook(feature_extraction_hook)

    features = {}
    for i, (input_data, paths) in enumerate(tqdm(loader)):
        input_var = torch.autograd.Variable(input_data, volatile=True).cuda()
        model(input_var)
        # current_features will be updated by feature_extraction_hook above.
        # Unfortunately there's no other general, straight-forward way to
        # extract features.
        for j, image_path in enumerate(paths):
            features[image_path] = current_features['features'][j]

    logging.info('Outputting features')
    with h5py.File(output_hdf5, 'a') as f:
        paths = features.keys()
        features_stacked = np.vstack([features[path] for path in paths])
        f.create_dataset('features', data=features_stacked)
        f.create_dataset(
            'image_names',
            data=[image_path_to_name(path) for path in paths],
            dtype=h5py.special_dtype(vlen=str))


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--image-list', help='Text file with path to image on each line.')
    parser.add_argument(
        '--arch-layer',
        default='alexnet-fc7',
        choices=[
            '{}-{}'.format(model, layer)
            for model, layers in get_layer.items() for layer in layers
        ],
        help='Architecture + layer to extract features from. Choices: ' +
        ', '.join([
            '{model}-{{{layers}}}'.format(
                model=model, layers=','.join(layers.keys()))
            for model, layers in get_layer.items()
        ]))
    parser.add_argument(
        '--output-features',
        metavar='PATH',
        type=str,
        required=True,
        help='Output features as HDF5 to this location.')
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help='number of data loading workers')
    parser.add_argument(
        '--batch-size',
        default=256,
        type=int,
        metavar='N')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument(
        '--pretrained', action='store_true', help='Use pre-trained model')

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    architecture, layer = args.arch_layer.split('-')
    model = models.__dict__[architecture](pretrained=args.pretrained)

    # Multi-GPU currently not supported.
    # if (architecture.startswith('alexnet')
    #         or architecture.startswith('vgg')):
    #     # Copied from
    #     # https://github.com/pytorch/examples/blob/d5678bc8ac0cdd79dbd5e44d4130271018bcec4e/imagenet/main.py
    #     model.features = torch.nn.DataParallel(model.features)
    # else:
    #     model = torch.nn.DataParallel(model)

    model.cuda()
    model.eval()

    layer = get_layer[architecture][layer](model)

    with open(args.image_list, 'r') as f:
        images = [line.strip() for line in f]

    extract_features_to_disk(images, model, layer, args.batch_size,
                             args.workers, args.output_features)


if __name__ == "__main__":
    main()
