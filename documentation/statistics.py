import csv
import sys

import numpy as np


def valid_data(data):
    valid = np.where(data[:, 0] > 0)
    invalid = np.where(data[:, 0] <= 0)
    return valid, invalid


def steps(data, mask):
    return np.min(data[mask, 1]), \
           np.median(data[mask, 1]), \
           np.max(data[mask, 1])


def score(data, mask):
    return np.min(data[mask, 0]), \
           np.median(data[mask, 0]), \
           np.max(data[mask, 0])


def ratio(data, mask):
    rat = data[mask, 0] / data[mask, 1]
    return np.min(rat), np.mean(rat), np.max(rat)


def print_summary(data, valid, invalid):
    fmt = '& ${}$ ' * 2 + '& ${:.2f}:{:.2f}:{:.2f}$ ' * 3
    out = fmt.format(data[valid].shape[0],
                     data[invalid].shape[0],
                     *steps(data, valid),
                     *score(data, valid),
                     *ratio(data, valid))
    print(out)


def compute_statistics(data):
    valid, invalid = valid_data(data)
    print_summary(data, valid, invalid)


if __name__ == '__main__':
    try:
        fname = sys.argv[sys.argv.index('--file') + 1]
    except (IndexError, AttributeError):
        sys.exit('Please provide a filename (--file [fname])!')

    with open(fname, 'r') as f:
        reader = csv.reader(f)
        data = np.array([[score, steps] for score, steps in reader],
                        dtype=np.float)

    try:
        compute_statistics(data)
    except TypeError as e:
        print(e)
        sys.exit('No data available.')
