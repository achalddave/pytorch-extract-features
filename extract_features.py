"""Extract features from a list of images.

Example usage:
    python extract_features.py \
        --image-list <(ls /path/to/images/*.jpg) \
        --arch-layer alexnet-fc7 \
        --output-features features.h5 \
        --batch-size 10 \
        --pretrained True

This will output an HDF5 file with two datasets: 'features' and 'image_names'.
The 'image_names' dataset will contain a list of length (num_images, ) that
contains the name of each image from the image list. The 'features' dataset
contains a list of length (num_images, num_features) that contains the features
for each image.
"""

import argparse
import logging
import random
import sys
from os import path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(
        models.__dict__[name]))


def load_model(model, model_file):
    checkpoint = torch.load(model_file)
    # Support for checkpoints saved by scripts based off of
    #   https://github.com/pytorch/examples/blob/master/imagenet/main.py
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    logging.info('Loading model from %s', model_file)
    model.load_state_dict(checkpoint, strict=False)

    missing_keys = set(model.state_dict().keys()) - set(checkpoint.keys())
    extra_keys = set(checkpoint.keys()) - set(model.state_dict().keys())
    if missing_keys:
        logging.info('Missing keys in --model-file: %s.', missing_keys)
    if extra_keys:
        logging.info('Extra keys ignored in --model-file: %s.', extra_keys)


class AlexNetPartial(nn.Module):
    supported_layers = [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'relu6', 'fc7',
        'fc8'
    ]

    def __init__(self,
                 layer,
                 model_file=None,
                 data_parallel=False,
                 **kwargs):
        super(AlexNetPartial, self).__init__()
        assert(layer in AlexNetPartial.supported_layers)
        self.model = models.alexnet(**kwargs)

        self.output_layer = layer

        if 'conv' in self.output_layer:
            # Map, e.g., 'conv2' to corresponding index into self.features
            conv_map = {}
            conv_index = 1
            for i, layer in enumerate(self.model.features):
                if isinstance(layer, nn.Conv2d):
                    conv_map['conv%s' % conv_index] = i
                    conv_index += 1
            requested_index = conv_map[self.output_layer]
            features = list(self.model.features.children())[:requested_index+1]
            self.model.features = nn.Sequential(*features)
        else:
            classifier_map = {
                'fc6': 1,
                'relu6': 2,
                'fc7': 4,
                'relu7': 5,
                'fc8': 6
            }
            requested_index = classifier_map[self.output_layer]
            classifier = list(
                self.model.classifier.children())[:requested_index+1]
            self.model.classifier = nn.Sequential(*classifier)
        if data_parallel:
            self.model.features = torch.nn.DataParallel(self.model.features)
        if model_file is not None:
            load_model(self.model, model_file)

    def forward(self, x):
        x = self.model.features(x)
        if 'conv' in self.output_layer:
            return x
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.model.classifier(x)
        return x


class VggPartial(nn.Module):
    supported_layers = ['fc1', 'relu1', 'fc2', 'relu2', 'fc3']

    def __init__(self,
                 architecture,
                 layer,
                 model_file=None,
                 data_parallel=False,
                 **kwargs):
        super(VggPartial, self).__init__()
        assert layer in VggPartial.supported_layers
        assert architecture.startswith('vgg')
        self.model = models.__dict__[architecture](**kwargs)
        self.output_layer = layer
        keep_upto = {
            'fc1': 0,
            'relu1': 1,
            'fc2': 3,
            'relu2': 4,
            'fc3': 6}[layer]
        classifier = list(
            self.model.classifier.children())[:keep_upto + 1]
        self.model.classifier = nn.Sequential(*classifier)
        if data_parallel:
            self.model.features = torch.nn.DataParallel(self.model.features)
        if model_file is not None:
            load_model(self.model, model_file)

    def forward(self, x):
        return self.model.forward(x)


class DenseNetPartial(nn.Module):
    supported_layers = ['avg_last', 'final']

    def __init__(self,
                 architecture,
                 layer,
                 model_file=None,
                 data_parallel=False,
                 **kwargs):
        super(DenseNetPartial, self).__init__()
        assert architecture.startswith('densenet')
        self.model = models.__dict__[architecture](**kwargs)
        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)
        if model_file is not None:
            load_model(self.model, model_file)
        self.output_layer = layer

    def forward(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(
            out, kernel_size=7, stride=1).view(features.size(0), -1)
        if self.output_layer == 'avg_last':
            return out
        elif self.output_layer == 'final':
            out = self.model.classifier(out)
            return out
        else:
            raise NotImplementedError


partial_models = {
    'alexnet': AlexNetPartial,
    'densenet121': DenseNetPartial,
    'densenet161': DenseNetPartial,
    'densenet169': DenseNetPartial,
    'densenet201': DenseNetPartial,
    'vgg11': VggPartial,
    'vgg11_bn': VggPartial,
    'vgg13': VggPartial,
    'vgg13_bn': VggPartial,
    'vgg16': VggPartial,
    'vgg16_bn': VggPartial,
    'vgg19_bn': VggPartial,
    'vgg19': VggPartial,
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
    # return np.string_(path.splitext(path.basename(image_path))[0])
    parent, image_name = path.split(image_path)
    image_name = path.splitext(image_name)[0]
    parent = path.split(parent)[1]
    return path.join(parent, image_name)


def extract_features_to_disk(image_paths,
                             model,
                             batch_size,
                             workers,
                             output_hdf5):
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = ListDataset(image_paths,
                          transforms.Compose([
                              transforms.Resize(256),
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

    features = {}
    for i, (input_data, paths) in enumerate(tqdm(loader)):
        input_var = torch.autograd.Variable(input_data, volatile=True).cuda()
        current_features = model(input_var).data.cpu().numpy()
        for j, image_path in enumerate(paths):
            features[image_path] = current_features[j]

    feature_shape = features[list(features.keys())[0]].shape
    logging.info('Feature shape: %s' % (feature_shape, ))
    logging.info('Outputting features')

    if sys.version_info >= (3, 0):
        string_type = h5py.special_dtype(vlen=str)
    else:
        string_type = h5py.special_dtype(vlen=unicode)  # noqa
    paths = features.keys()
    logging.info('Stacking features')
    features_stacked = np.vstack([features[path] for path in paths])
    logging.info('Output feature size: %s' % (features_stacked.shape, ))
    with h5py.File(output_hdf5, 'a') as f:
        f.create_dataset('features', data=features_stacked)
        f.create_dataset(
            'image_names',
            (len(paths), ),
            dtype=string_type)
        # For some reason, assigning the list directly causes an error, so we
        # assign it in a loop.
        for i, image_path in enumerate(paths):
            f['image_names'][i] = image_path_to_name(image_path)


def parse_bool(arg):
    """Parse string to boolean.

    Using type=bool in argparse does not do the right thing. E.g.
    '--bool_flag False' will parse as True. See
    <https://stackoverflow.com/q/15008758/1291812>
    """
    if arg == 'True':
        return True
    elif arg == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError("Expected 'True' or 'False'.")


def _set_logging(logging_filepath):
    """Setup logger to log to file and stdout."""
    log_format = '%(asctime)s.%(msecs).03d: %(message)s'
    date_format = '%H:%M:%S'

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(logging_filepath)
    file_handler.setFormatter(
        logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console_handler)

    logging.info('Writing log file to %s', logging_filepath)


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
            '{}-{}'.format(name, layer)
            for name, partial in partial_models.items()
            for layer in partial.supported_layers
        ],
        help='Architecture + layer to extract features from. Choices: ' +
        ', '.join([
            '{model}-{{{layers}}}'.format(
                model=model, layers=','.join(partial.supported_layers))
            for model, partial in partial_models.items()
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
        '--pretrained',
        type=parse_bool,
        default=True,
        help='Whether to use pre-trained model')
    parser.add_argument(
        '--data-parallel',
        type=parse_bool,
        default=True,
        help='Whether to use torch.nn.DataParallel')

    parser.add_argument(
        '--model-file',
        help='Load model weights from a file.')

    parser.add_argument(
        '--output_log',
        help='Output file to log to. Default: --output_features + ".log"')

    args = parser.parse_args()

    assert not path.exists(args.output_features)
    if args.model_file is not None:
        assert not args.pretrained, (
            '--pretrained cannot be specified if --model_file is specified.')
    if args.output_log is None:
        args.output_log = args.output_features + '.log'
    _set_logging(args.output_log)
    logging.info('Args: %s', args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    architecture, layer = args.arch_layer.split('-')

    construction_kwargs = {
        'layer': layer,
        'pretrained': args.pretrained,
        'model_file': args.model_file,
        'data_parallel': args.data_parallel,
    }
    if architecture.startswith('densenet') or architecture.startswith('vgg'):
        construction_kwargs['architecture'] = architecture
    model = partial_models[architecture](**construction_kwargs)

    model.cuda()
    model.eval()

    with open(args.image_list, 'r') as f:
        images = [line.strip() for line in f]

    extract_features_to_disk(images, model, args.batch_size,
                             args.workers, args.output_features)


if __name__ == "__main__":
    main()
