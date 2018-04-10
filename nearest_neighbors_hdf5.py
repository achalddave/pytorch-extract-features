"""Example usage of extracted features."""

import argparse
import random

import h5py
import numpy as np


def main():
    # Use first line of file docstring as description if a file docstring exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('features_hdf5')

    args = parser.parse_args()

    with h5py.File(args.features_hdf5, 'r') as f:
        # (num_samples, num_dimensions) matrix
        print('Loading features, shape: %s' % (f['features'].shape, ))
        features = np.array(f['features'])
        # List of length (num_samples)
        names = list(f['image_names'])
    print('Loaded features')

    # Sample a few data points, compute their nearest neighbors.
    num_sample = 5
    num_neighbors = 3
    sampled = list(range(len(features)))
    random.shuffle(sampled)
    sampled = sampled[:num_sample]

    for sample_index in sampled:
        feature = features[sample_index]
        other_features = np.vstack(
            [features[:sample_index], features[sample_index + 1:]])
        distances = np.linalg.norm(other_features - feature.T, axis=1, ord=2)
        neighbors = np.argsort(distances)[:num_neighbors]
        print('Neighbors of {}: {}'.format(
            names[sample_index], ', '.join(names[i] for i in neighbors)))


if __name__ == "__main__":
    main()
