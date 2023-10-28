import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from natsort import natsorted
from tqdm import tqdm

from asclepius import dataset
from asclepius.config import logger

app = typer.Typer()


@app.command()
def prepare_dataset(
    dataset_root: Path = Path("dataset").resolve(),
    metadata_file: Optional[str] = "metadata",
):
    shutil.rmtree(dataset_root.joinpath("preprocessed"))
    logger.info(f"loading dataset...")

    metadata = dataset.load_metadata(dataset_root, metadata_file)
    logger.info(f"found metadata: {metadata}")

    patient_directories = dataset.get_patient_directories(dataset_root)
    patient_ids = dataset.get_patient_ids(patient_directories)
    logger.info(f"found patient IDs: {patient_ids}")

    assert natsorted(metadata.keys()) == natsorted(patient_ids)
    logger.info(f"patient IDs and directories match!")

    logger.info(f"processing patient data...")
    for patient_id in tqdm(patient_ids):
        patient_metadata = metadata[patient_id]
        mat_files = dataset.load_patient_data(
            dataset_root, patient_id, patient_metadata
        )
        transformed_patient_data = dataset.prepare_patient_data(
            mat_files, patient_metadata
        )
        dataset.label_and_save_frames(
            dataset_root, patient_id, transformed_patient_data, patient_metadata
        )
    logger.info(
        f"patient data preprocessed and saved to: {dataset_root.joinpath('preprocessed')}"
    )
    return


if __name__ == "__main__":
    app()
