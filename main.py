from datetime import datetime
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import typer
from natsort import natsorted
from tqdm import tqdm
from typing_extensions import Annotated

from asclepius import data, train
from asclepius.config import logger

app = typer.Typer()


@app.command()
def prepare_dataset(
    split: Annotated[str, typer.Option()],
    dataset_root: Path = Path("dataset").resolve(),
    metadata_file: Annotated[str, typer.Option()] = "metadata",
):
    target_dir = dataset_root.joinpath(f"preprocessed/{split}")
    if Path.exists(target_dir):
        logger.info(f"found existing data at: {target_dir}")
        logger.info("cleaning...")
        shutil.rmtree(target_dir)

    logger.info(f"loading dataset...")

    metadata = data.load_metadata(dataset_root, metadata_file)
    logger.info(f"found metadata: {metadata}")

    patient_directories = data.get_patient_directories(dataset_root)
    patient_ids = data.get_patient_ids(patient_directories)
    logger.info(f"found patient IDs: {patient_ids}")

    assert natsorted(metadata.keys()) == natsorted(patient_ids)
    logger.info(f"patient IDs and directories match!")

    logger.info(f"processing patient data...")
    for patient_id in tqdm(patient_ids):
        patient_metadata = metadata[patient_id]
        mat_files = data.load_patient_data(dataset_root, patient_id, patient_metadata)
        transformed_patient_data = data.prepare_patient_data(
            mat_files, patient_metadata
        )
        data.label_and_save_frames(
            dataset_root, split, patient_id, transformed_patient_data, patient_metadata
        )
    logger.info(
        f"patient data preprocessed and saved to: {dataset_root.joinpath('preprocessed')}"
    )
    return


@app.command()
def train_val_test_split(
    dataset_split_path: Annotated[Path, typer.Option()],
    split_ratios: Annotated[str, typer.Option()],
    patient_id_split_json: Annotated[Path, typer.Option()] = None,
):
    split_ratios = split_ratios.split(",")
    split_ratios = [int(split_ratio) for split_ratio in split_ratios]
    if patient_id_split_json is not None:
        with open(patient_id_split_json, "r") as jf:
            patient_id_split = json.load(jf)
    else:
        patient_id_split = None
    data.split_dataset(dataset_split_path, split_ratios, patient_id_split)


@app.command()
def trainer(
    model_names: Annotated[str, typer.Option()] = [
        "cnn",
        "densenet",
        "vit",
        "swin",
    ],
    output_dir: Annotated[Path, typer.Option()] = Path(f"output").resolve(),
    dataset_root: Annotated[Path, typer.Option()] = Path(
        "dataset/preprocessed"
    ).resolve(),
    split: Annotated[str, typer.Option()] = "bal",
    modalities: Annotated[str, typer.Option()] = "US",
    target_size: Annotated[str, typer.Option()] = "150,390",
    num_classes: Annotated[int, typer.Option()] = 2,
    batch_size: Annotated[int, typer.Option()] = 32,
    shuffle: Annotated[bool, typer.Option()] = True,
    learning_rate: Annotated[float, typer.Option()] = 1e-3,
    num_epochs: Annotated[int, typer.Option()] = 100,
    wd: Annotated[float, typer.Option()] = 1e-4,
    dropout: Annotated[float, typer.Option()] = 0.5,
    early_stopping_patience: Annotated[int, typer.Option()] = 10,
):
    model_names = model_names.split(",")
    logger.info(f"model names: {model_names}")

    modalities = modalities.split(",")
    logger.info(f"modalities: {modalities}")

    target_size = target_size.split(",")
    target_size = [int(size) for size in target_size]

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    mode = "train"

    if not Path.exists(output_dir):
        logger.info(f"creating parent output directory: {output_dir}")
        Path.mkdir(output_dir)
    target_dir = output_dir.joinpath(timestamp)
    Path.mkdir(target_dir)
    logger.info(f"created output directory: {target_dir}")

    for modality in modalities:
        logger.info(f"training for modality: {modality}")
        for model_name in model_names:
            logger.info(f"training model: {model_name}")

            use_dct = True if "DCT" in modality else False
            logger.info(f"using DCT: {use_dct}")

            use_gabor = True if "Gabor" in modality else False
            logger.info(f"using Gabor: {use_gabor}")

            current_target_dir = target_dir.joinpath(f"{model_name}/{modality}/")
            Path.mkdir(current_target_dir, parents=True)

            if "Dual" in modality:
                in_channels = 4
            elif "PAUS" in modality:
                in_channels = 2
            else:
                in_channels = 1
            logger.info(f"input channels: {in_channels}")

            train_loader, val_loader = train.get_dataloaders(
                dataset_root,
                split,
                mode,
                modality,
                target_size,
                num_classes,
                use_dct,
                use_gabor,
                batch_size,
                shuffle,
            )
            train.train_model(
                model_name,
                in_channels,
                current_target_dir,
                train_loader,
                val_loader,
                batch_size,
                learning_rate,
                num_epochs,
                wd,
                dropout,
                early_stopping_patience,
                target_size,
            )


if __name__ == "__main__":
    app()
