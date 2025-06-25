"""Extract features for raw IFCB data"""

import os
from multiprocessing import get_context
from pathlib import Path

from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass

import numpy as np
from ifcb_features import compute_features
from sykepic.utils import files, ifcb, logger

VERSION = "py-v4"
FILE_SUFFIX = ".feat"
log = logger.get_logger("feat")


def call(args):
    """
    Process command line arguments and extract features from samples.
    Args:
        args: Command line arguments containing sample paths, output directory,
                parallel processing flag, and force overwrite flag.
    Returns:
        None
    """
    if args.raw:
        sample_paths: List[Path] = files.list_sample_paths(args.raw)
        filtered_sample_paths = filter_sample_paths(sample_paths)
        process_sample_list(
            sample_paths=filtered_sample_paths,
            sample_type="ifcb",
            out_dir=args.out,
            parallel=args.parallel,
            force=args.force,
        )
    elif args.samples:
        sample_paths: List[Path] = [Path(path) for path in args.samples]
        filtered_sample_paths = filter_sample_paths(sample_paths)
        process_sample_list(
            sample_paths=filtered_sample_paths,
            sample_type="ifcb",
            out_dir=args.out,
            parallel=args.parallel,
            force=args.force,
        )
    else: # args.image_dir
        pass

def filter_sample_paths(sample_paths: List[Path]) -> List[Path]:
    """Filter sample paths to exclude those with .roi files larger than 1GB.
    Args:
        sample_paths (List[Path]): List of sample paths to filter.
    Returns:
        List[Path]: Filtered list of sample paths.
    """
    filtered_sample_paths = []
    for sample_path in sample_paths:
        if sample_path.with_suffix(".roi").stat().st_size <= 1e9:
            filtered_sample_paths.append(sample_path)
        else:
            log.warn(f"{sample_path.name} is over 1G, skipping")
    return filtered_sample_paths

def process_sample_list(
    sample_paths: List[Path],
    sample_type: Literal["ifcb", "img"],
    out_dir: str,
    parallel: bool = False,
    force: bool = False
):
    """
    Extract features from a list of sample paths and save them to CSV files.
    Handles parallelization of process_sample processing.

    Args:
        sample_paths (List[Path]): List of paths to IFCB sample files.
        sample_type (Literal["ifcb", "img"]): Type of sample being processed.
        out_dir (str): Output directory where the CSV files will be saved.
        parallel (bool): Whether to process samples in parallel.
        force (bool): Whether to overwrite existing CSV files.
    Returns:
        set: A set of sample names that were processed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if parallel:
        available_cores = os.cpu_count()
        log.debug(f"Extracting features in parallel with {available_cores} cores")
        with get_context("spawn").Pool(available_cores) as pool:
            samples_processed = pool.starmap(
                process_sample, [(path, sample_type, out_dir, force) for path in sample_paths]
            )
    else:
        log.debug("Extracting features synchronously")
        samples_processed = []
        for path in sorted(sample_paths):
            samples_processed.append(process_sample(path, sample_type, out_dir, force))

    if sample_type == "img":
        # Placeholder for image directory sample aggregation
        pass
    return set(filter(None, samples_processed))


def process_sample(sample_path: Path,
                   sample_type: Literal["ifcb", "img"],
                   out_dir: str,
                   force: bool=False):
    """
    Process a single sample to extract features and save them to a CSV file.
    Handles the creation of the output directory and checks for existing files.

    For IFCB samples that contain multiple ROIs, it extracts features from each ROI
    and saves them in a CSV file.
    
    For image samples, 

    Args:
        sample_path (Path): Path to the sample file.
        sample_type (Literal["ifcb", "img"]): Type of sample being processed.
        out_dir (Path): Output directory where the CSV file will be saved.
        force (bool): Whether to overwrite existing CSV files.
    Returns:
        str: The name of the processed sample, or None if it was skipped.
    """
    pid = os.getpid()

    csv_path: Path = files.sample_csv_path(sample_path, out_dir, suffix=FILE_SUFFIX)
    if csv_path.is_file():
        if force:
            print(
                f"feat [{pid}] - WARNING - {csv_path.name} already exists, overwriting"
            )
        else:
            print(f"feat [{pid}] - WARNING - {csv_path.name} already exists, skipping")
            return sample_path.name

    print(f"feat [{pid}] - INFO - Extracting features for {sample_path.name}")

    if sample_type == 'ifcb':
        roi_features = sample_ifcb_features(sample_path)
        ifcb_features_to_csv(roi_features, csv_path)
    elif sample_type == 'img':
        pass
    else:
        raise ValueError(f"Unknown sample type: {sample_type}")

    return sample_path.name

@dataclass
class ROIFeatures:
    roi_id: str
    sample_type: str
    biovol_px: int
    area: float
    major_axis_length: float
    minor_axis_length: float
    biovol_um3: Optional[float] = None
    biomass_ugl: Optional[float]= None
    volume_ml: Optional[float] = None

def sample_ifcb_features(sample_path: Path) -> List[ROIFeatures]:
    root = Path(sample_path)
    adc = root.with_suffix(".adc")
    hdr = root.with_suffix(".hdr")
    roi = root.with_suffix(".roi")
    try:
        volume_ml = sample_volume(hdr)
        # This didn't show in logs for some reason
        # if volume_ml <= 0:
        #   log.warn(f"{root.name} volume_ml is {volume_ml}")
    except Exception:
        log.exception(f"Unable to calculate volume for {root.name}")
        return None
    roi_features = []
    for roi_id, roi_array in ifcb.raw_to_numpy(adc, roi):
        roi_features.append(calculate_roi_features(roi_id=roi_id,
                                                   sample_type='ifcb',
                                                   roi_array=roi_array,
                                                   volume_ml=volume_ml))
    return roi_features

def calculate_roi_features(roi_id: str,
                           sample_type: Literal['ifcb', 'img'],
                           roi_array: np.array,
                           volume_ml: Optional[float] = None) -> ROIFeatures:
    """
    Calculate features for a single ROI array.
    Args:
        roi_id (str): Identifier for the ROI.
        sample_type (Literal['ifcb', 'img']): Type of sample being processed. Is saved in the features.
        roi_array (np.array): Numpy array representing the ROI.
        volume_ml (Optional[float]): Volume of the sample in milliliters. If None, some features will be None.
    Returns:
        ROIFeatures: A dataclass containing the calculated features for the ROI.
    """
    _, all_roi_features = compute_features(roi_array)
    all_roi_features = dict(all_roi_features)

    biovol_px = all_roi_features["Biovolume"]
    area = all_roi_features["Area"]
    major_axis_length = all_roi_features["MajorAxisLength"]
    minor_axis_length = all_roi_features["MinorAxisLength"]

    if volume_ml is not None:
        biovol_um3 = pixels_to_um3(biovol_px)
        biomass_ugl = biovolume_to_biomass(biovol_um3, volume_ml)
    else:
        biovol_um3 = None
        biomass_ugl = None

    return ROIFeatures(
        roi_id=roi_id,
        sample_type=sample_type,
        biovol_px=biovol_px,
        biovol_um3=biovol_um3,
        biomass_ugl=biomass_ugl,
        area=area,
        major_axis_length=major_axis_length,
        minor_axis_length=minor_axis_length,
        volume_ml=volume_ml
    )


def sample_volume(hdr_file):
    ifcb_flowrate = 0.25  # ml
    run_time = None
    inhibit_time = None
    with open(hdr_file) as fh:
        for line in fh:
            if line.startswith("inhibitTime"):
                inhibit_time = float(line.split()[1])
            elif line.startswith("runTime"):
                run_time = float(line.split()[1])
    sample_vol = ifcb_flowrate * ((run_time - inhibit_time) / 60.0)
    if sample_vol <= 0:
        raise ValueError(f"Sample volume is {sample_vol}")
    return sample_vol


def pixels_to_um3(pixels, micron_factor=2.8):
    return pixels / (micron_factor**3)
    # The micron factor should be 2.8 for python features v-4, but should still be tested for instruments


def biovolume_to_biomass(biovol_um3, volume_ml):
    try:
        return biovol_um3 / volume_ml / 1000
    except ZeroDivisionError:
        return 0


def ifcb_features_to_csv(roi_features: List[ROIFeatures], csv_path: str):
    sample_types = set([f.sample_type for f in roi_features])
    if sample_types != {'ifcb'}:
        raise ValueError(f"All ROI features must be of type 'ifcb'. Now they are {sample_types}")

    volume_mls = set([f.volume_ml for f in roi_features])
    if len(volume_mls) != 1:
        raise ValueError(
            f"All ROI features must have the same volume_ml. Now they are {volume_mls}"
        )
    volume_ml = roi_features[0].volume_ml
    selected_features = [
        (
            roi_feat.roi_id,
            roi_feat.biovol_px,
            roi_feat.biovol_um3,
            roi_feat.biomass_ugl,
            roi_feat.area,
            roi_feat.major_axis_length,
            roi_feat.minor_axis_length
        )
        for roi_feat in roi_features
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # csv_content = f"# {datetime.now().astimezone().isoformat()}\n"
    csv_content = f"# version={VERSION}\n"
    csv_content += f"# volume_ml={volume_ml}\n"
    csv_content += (
        "roi,biovolume_px,biovolume_um3,biomass_ugl,"
        "area,major_axis_length,minor_axis_length\n"
    )
    for roi_feat in selected_features:
        csv_content += ",".join(map(str, roi_feat)) + "\n"
    with open(csv_path, "w") as fh:
        fh.write(csv_content)

