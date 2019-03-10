import os
import sys
import random
import numpy as np

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)
from utils.LungsLoader import LungsLoader


loader = LungsLoader()
identity_affine = np.array([[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]])
# TODO : add batchsize


def atlas_generator(atlas_id, scans_ids, width, height, depth, loop=False, shuffle=False, use_affine=True):
    """Iterates over scans with single target atlas scan to registrate

    Args:
        atlas_id (str): atlas id
        scans_ids (list): list of scans ids to iterate over
        width (int): desired output width
        height (int): desired output height
        depth (int): desired output depth
        loop (boolean): if true, loops indefinitely over scans_ids (default: False)
        shuffle (boolean): if true, shuffles scans_ids before each new loop (default: False)
        use_affine (boolean): if true, yield affine identity transformation
    """
    atlas_scan_gen = loader.preprocess_scans([atlas_id], width, height, depth)
    atlas_scan = next(atlas_scan_gen)[0][np.newaxis, :, :, :, np.newaxis]
    scan_gen = loader.preprocess_scans(scans_ids, width, height, depth, loop=loop, shuffle=shuffle)
    identity_flow = 0.5 * np.ones((1,) + (width, height, depth) + (3,))
    try:
        while True:
            src_scan = next(scan_gen)[0]
            src_scan = src_scan[np.newaxis, :, :, :, np.newaxis]
            if use_affine:
                yield ([src_scan, atlas_scan], [atlas_scan, identity_flow, identity_affine])
            else:
                yield ([src_scan, atlas_scan], [atlas_scan, identity_flow])
    except StopIteration:
        raise StopIteration(f"Completed iteration over the f{len(scans_ids)} scans")


def scan_generator(scans_ids, width, height, depth, loop=False, shuffle=False, use_affine=True):
    """Iterates over luna dataset yielding scans

    Args:
        scans_ids (list): list of scans ids to iterate over
        width (int): desired output width
        height (int): desired output height
        depth (int): desired output depth
        loop (boolean): if true, loops indefinitely over scans_ids (default: False)
        shuffle (boolean): if true, shuffles scans_ids before each new loop (default: False)
        use_affine (boolean): if true, yield affine identity transformation
    """
    scan_gen = loader.preprocess_scans(scans_ids, width, height, depth, loop=loop, shuffle=shuffle)
    identity_flow = 0.5 * np.ones((1,) + (width, height, depth) + (3,))
    try:
        while True:
                src_scan = next(scan_gen)[0]
                tgt_scan = next(scan_gen)[0]
                src_scan = src_scan[np.newaxis, :, :, :, np.newaxis]
                tgt_scan = tgt_scan[np.newaxis, :, :, :, np.newaxis]
                if use_affine:
                    yield ([src_scan, tgt_scan], [tgt_scan, identity_flow, identity_affine])
                else:
                    yield ([src_scan, tgt_scan], [tgt_scan, identity_flow])
    except StopIteration:
        raise StopIteration(f"Completed iteration over the f{len(scans_ids)} scans")


def scan_and_seg_generator(scans_ids, width, height, depth, loop=False, shuffle=False, use_affine=True):
    """Iterates over luna dataset yielding scans and their segmentation

    Args:
        scans_ids (list): list of scans ids to iterate over
        width (int): desired output width
        height (int): desired output height
        depth (int): desired output depth
        loop (boolean): if true, loops indefinitely over scans_ids (default: False)
        shuffle (boolean): if true, shuffles scans_ids before each new loop (default: False)
        use_affine (boolean): if true, yield affine identity transformation
    """
    identity_flow = 0.5 * np.ones((1,) + (width, height, depth) + (3,))
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

                if use_affine:
                    yield ([src_scan, tgt_scan, src_seg], [tgt_scan, tgt_seg, identity_flow, identity_affine])
                else:
                    yield ([src_scan, tgt_scan, src_seg], [tgt_scan, tgt_seg, identity_flow])
        except StopIteration:
            if loop:
                continue
            else:
                raise StopIteration(f"Completed iteration over the f{len(scans_ids)} scans")


def atlas_seg_generator(atlas_id, scans_ids, width, height, depth, loop=False, shuffle=False, use_affine=True):
    """Iterates over luna dataset yielding scans and their segmentation

    Args:
        scans_ids (list): list of scans ids to iterate over
        width (int): desired output width
        height (int): desired output height
        depth (int): desired output depth
        loop (boolean): if true, loops indefinitely over scans_ids (default: False)
        shuffle (boolean): if true, shuffles scans_ids before each new loop (default: False)
        use_affine (boolean): if true, yield affine identity transformation
    """
    atlas_scan_gen = loader.preprocess_scans([atlas_id], width, height, depth)
    atlas_scan = next(atlas_scan_gen)[0][np.newaxis, :, :, :, np.newaxis]
    atlas_seg_gen = loader.preprocess_segmentations([atlas_id], width, height, depth)
    atlas_seg = next(atlas_seg_gen)[0][np.newaxis, :, :, :, np.newaxis]
    identity_flow = 0.5 * np.ones((1,) + (width, height, depth) + (3,))
    while True:
        if shuffle:
            random.shuffle(scans_ids)
        scan_gen = loader.preprocess_scans(scans_ids, width, height, depth, loop=True, shuffle=False)
        seg_gen = loader.preprocess_segmentations(scans_ids, width, height, depth, loop=True, shuffle=False)

        try:
            while True:
                src_scan = next(scan_gen)[0]
                src_scan = src_scan[np.newaxis, :, :, :, np.newaxis]

                src_seg = next(seg_gen)[0]
                src_seg = src_seg[np.newaxis, :, :, :, np.newaxis]

                if use_affine:
                    yield ([src_scan, atlas_scan, src_seg], [atlas_scan, identity_flow, identity_affine, atlas_seg])
                else:
                    yield ([src_scan, atlas_scan, src_seg], [atlas_scan, identity_flow, atlas_seg])
        except StopIteration:
            if loop:
                continue
            else:
                raise StopIteration(f"Completed iteration over the f{len(scans_ids)} scans")
