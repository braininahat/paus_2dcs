import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import h5py
import numpy as np
import scipy.io as scio
import torch
from natsort import natsorted
from scipy.fftpack import dct
from skimage.exposure import rescale_intensity
from torch.utils.data import Dataset
from torchvision import transforms


class BreastDataset(Dataset):
    def __init__(
        self,
        data_dir,
        modalities=["PA", "US"],
        target_size=(150, 390),
        num_classes=2,
        use_dct=False,
        use_gabor=False,
    ):
        # TODO add GLCM
        self.data_dir = data_dir
        self.modalities = modalities
        self.target_size = target_size
        self.num_classes = num_classes

        # Define the parameters for the Gabor filter
        ksize = 31  # Size of the filter kernel
        sigma = 5  # Standard deviation of the Gaussian
        theta = np.pi / 2  # Orientation of the Gabor filter (0 degrees)
        lambd = 10  # Wavelength of the sinusoidal factor
        gamma = 0.5  # Spatial aspect ratio
        psi = 0.0  # Phase offset

        # Create the Gabor kernel
        self.gabor_kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, psi
        )
        self.gabor_lambda = lambda x: cv2.filter2D(
            np.array(x), cv2.CV_8UC3, self.gabor_kernel
        )
        self.dct_lambda = lambda x: dct(
            dct(x, axis=0, norm="ortho"), axis=1, norm="ortho"
        )
        transforms_list = [
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ]

        if "PA" in modalities:
            self.pa_file_list = natsorted(glob(f"{self.data_dir}/*PA*.mat"))
        if "US" in modalities:
            self.us_file_list = natsorted(glob(f"{self.data_dir}/*US*.mat"))
        if "PA" in modalities and "US" in modalities:
            assert len(self.pa_file_list) == len(self.us_file_list)

        if use_dct and use_gabor:
            raise ValueError("Only one of DCT or Gabor should be True.")
        if use_dct:
            transforms_list.insert(2, transforms.Lambda(self.dct_lambda))
        if use_gabor:
            transforms_list.insert(2, transforms.Lambda(self.gabor_lambda))

        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        if "PA" in self.modalities:
            return len(self.pa_file_list)
        elif "US" in self.modalities:
            return len(self.us_file_list)

    def __getitem__(self, idx):
        if "PA" in self.modalities:
            pa_file_path = self.pa_file_list[idx]
            pa_data = scio.loadmat(pa_file_path)
            pa_frame = pa_data["frame"]
            pa_label = pa_data["label"]
        if "US" in self.modalities:
            us_file_path = self.us_file_list[idx]
            us_data = scio.loadmat(us_file_path)
            us_frame = us_data["frame"]
            us_label = us_data["label"]

        if "PA" in self.modalities and "US" in self.modalities:
            assert pa_label == us_label

        if "PA" in self.modalities:
            label = pa_label
        elif "US" in self.modalities:
            label = us_label

        # Convert label to a numeric value, e.g., 0 for 'clean' and 1 for 'tumor'
        label = 0 if label == "clean" else 1

        if "PA" in self.modalities:
            pa_frame = self.transform((pa_frame * 255).astype(np.uint8))
            pa_frame = torch.Tensor(pa_frame).float()
        if "US" in self.modalities:
            us_frame = self.transform((us_frame * 255).astype(np.uint8))
            us_frame = torch.Tensor(us_frame).float()

        if "PA" in self.modalities and "US" in self.modalities:
            frame = torch.cat([pa_frame, us_frame], axis=0).float()
            label = torch.tensor(label)
            return frame, label
        elif "PA" in self.modalities:
            return pa_frame, label
        elif "US" in self.modalities:
            return us_frame, label


def loadmat_custom(path):
    old = False
    try:
        mat = scio.loadmat(path)
    except NotImplementedError as e:
        old = True
        mat = h5py.File(path, "r")
    mat_keys = [matkey for matkey in mat.keys() if not matkey.startswith("_")]
    if len(mat_keys) > 1:
        retval = [mat[matkey][:] for matkey in mat_keys]
    else:
        retval = mat[mat_keys[0]][:]
    if old:
        mat.close()
    return retval


def load_patient_data(patient_id, dataset_root=Path("dataset/")):
    patient_dir = dataset_root.joinpath(patient_id)
    mat_files = natsorted(glob(f"{patient_dir}/*.mat"))

    assert len(mat_files) == 2

    pa_path, us_path = mat_files

    pa_mat = loadmat_custom(pa_path)
    us_mat = loadmat_custom(us_path)

    return pa_mat, us_mat


def get_patient_directories(dataset_root: Path):
    return natsorted([entry for entry in dataset_root.iterdir() if entry.is_dir()])


def load_metadata(dataset_root: Path, filename: Optional[str] = "metadata.json"):
    with open(dataset_root.joinpath(f"{filename}.json"), "r") as jf:
        return json.load(jf)


def get_patient_ids(directories: List[Path]):
    return [directory.stem for directory in directories]


def get_patient_metadata_pairs(patient_ids: List[str], metadata: Dict):
    return [(patient_id, metadata[patient_id]) for patient_id in patient_ids]


def load_patient_data(dataset_root: Path, patient_id: str, metadata: Dict):
    pa_mat = loadmat_custom(
        dataset_root.joinpath(patient_id, f"pa_{metadata['metadata']['malignant']}.mat")
    )
    us_mat = loadmat_custom(
        dataset_root.joinpath(patient_id, f"us_{metadata['metadata']['malignant']}.mat")
    )

    return {"pa_mat": pa_mat, "us_mat": us_mat}


def transform_arr(arr: np.ndarray, transform_metadata: Dict):
    # flip
    if transform_metadata["flip"] is not None and len(transform_metadata["flip"]) > 0:
        arr = np.flip(arr, axis=transform_metadata["flip"])

    # crop to tumor half
    if transform_metadata["half"] == "upper":
        arr = arr[:, : arr.shape[1] // 2, :]
    elif transform_metadata["half"] == "lower":
        arr = arr[:, arr.shape[1] // 2 :, :]
    else:
        raise ValueError(
            f"missing mat half information in metadata. found half value: {transform_metadata['half']}"
        )

    # trim empty space and edges to make both volumes match
    trim_limits = transform_metadata["trim"]
    trim_axis_0 = trim_limits["0"]
    trim_axis_1 = trim_limits["1"]
    trim_axis_2 = trim_limits["2"]
    arr = arr[
        trim_axis_0[0] : trim_axis_0[1],
        trim_axis_1[0] : trim_axis_1[1],
        trim_axis_2[0] : trim_axis_2[1],
    ]
    return arr


def prepare_patient_data(mat_files: Dict, metadata: Dict):
    pa_mat = mat_files["pa_mat"]
    pa_transforms = metadata["pa"]
    pa_arr = transform_arr(pa_mat, pa_transforms)

    us_mat = mat_files["us_mat"]
    us_transforms = metadata["us"]
    us_arr = transform_arr(us_mat, us_transforms)

    return {"pa_arr": pa_arr, "us_arr": us_arr}


def label_and_save_frames(
    data_root: Path, patient_id: str, transformed_arrs: Dict, metadata: Dict
):
    target_dir = data_root.joinpath("preprocessed")
    os.makedirs(target_dir, exist_ok=True)

    pa_arr_shape = transformed_arrs["pa_arr"].shape
    us_arr_shape = transformed_arrs["us_arr"].shape

    tumor_bounds = metadata["tumor_bounds"]["sides"]

    assert pa_arr_shape[1:] == us_arr_shape[1:]

    pa_norm = rescale_intensity(transformed_arrs["pa_arr"], out_range=(0, 1))
    us_norm = rescale_intensity(transformed_arrs["us_arr"], out_range=(0, 1))
    for cs_index in range(pa_arr_shape[-1]):
        if cs_index > tumor_bounds[0] and cs_index < tumor_bounds[1]:
            label = "tumor"
        else:
            label = "clean"
        scio.savemat(
            str(target_dir.joinpath(f"{patient_id}_PA_{cs_index}.mat")),
            {"frame": pa_norm[:, :, cs_index], "label": label},
        )
        scio.savemat(
            str(target_dir.joinpath(f"{patient_id}_US_{cs_index}.mat")),
            {"frame": us_norm[:, :, cs_index], "label": label},
        )
    return True
