import os
import sys
import random
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
from utils.LungsLoader import LungsLoader


loader = LungsLoader()
# TODO : add batchsize


def scan_generator(scans_ids, width, height, depth, loop=False, shuffle=False):
    """Iterates over luna dataset yielding scans

    Args:
        scans_ids (list): list of scans ids to iterate over
        width (int): desired output width
        height (int): desired output height
        depth (int): desired output depth
        loop (boolean): if true, loops indefinitely over scans_ids (default: False)
        shuffle (boolean): if true, shuffles scans_ids before each new loop (default: False)
    """
    scan_gen = loader.preprocess_scans(scans_ids, width, height, depth, loop=loop, shuffle=shuffle)
    zeros = np.zeros((1,) + (width, height, depth) + (3,))
    identity = np.array([[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]])
    try:
        while True:
                src_scan = next(scan_gen)[0]
                tgt_scan = next(scan_gen)[0]
                src_scan = src_scan[np.newaxis, :, :, :, np.newaxis]
                tgt_scan = tgt_scan[np.newaxis, :, :, :, np.newaxis]
                yield ([src_scan, tgt_scan], [tgt_scan, zeros, identity])
    except StopIteration:
        raise StopIteration(f"Completed iteration over the f{len(scans_ids)} scans")


def scan_and_seg_generator(scans_ids, width, height, depth, loop=False, shuffle=False):
    """Iterates over luna dataset yielding scans and their segmentation

    Args:
        scans_ids (list): list of scans ids to iterate over
        width (int): desired output width
        height (int): desired output height
        depth (int): desired output depth
        loop (boolean): if true, loops indefinitely over scans_ids (default: False)
        shuffle (boolean): if true, shuffles scans_ids before each new loop (default: False)
    """
    zeros = np.zeros((1,) + (width, height, depth) + (3,))
    identity = np.array([[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]])
    while True:
        if shuffle:
            random.shuffle(scans_ids)
        scan_gen = loader.preprocess_scans(scans_ids, width, height, depth, loop=True, shuffle=False)
        seg_gen = loader.preprocess_segmentations(scans_ids, width, height, depth, loop=True, shuffle=False)

        try:
            while True:
                src_scan = next(scan_gen)[0]
                tgt_scan = next(scan_gen)[0]
                src_scan = src_scan[np.newaxis, :, :, :, np.newaxis]
                tgt_scan = tgt_scan[np.newaxis, :, :, :, np.newaxis]

                src_seg = next(seg_gen)[0]
                tgt_seg = next(seg_gen)[0]
                src_seg = src_seg[np.newaxis, :, :, :, np.newaxis]
                tgt_seg = tgt_seg[np.newaxis, :, :, :, np.newaxis]

                yield ([src_scan, tgt_scan, src_seg], [tgt_scan, tgt_seg, zeros, identity])
        except StopIteration:
            if loop:
                continue
            else:
                raise StopIteration(f"Completed iteration over the f{len(scans_ids)} scans")
