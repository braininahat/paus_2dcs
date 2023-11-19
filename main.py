import json
import shutil
from datetime import datetime
from glob import glob
from pathlib import Path
from pyrsistent import b

import torch
import typer
import wandb
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
    buffer: Annotated[int, typer.Option()] = 10,
):
    target_dir = dataset_root.joinpath(f"preprocessed/{split}")
    if Path.exists(target_dir):
        logger.info(f"found existing data at: {target_dir}")
        logger.info("cleaning...")
        shutil.rmtree(target_dir)

    logger.info(f"loading dataset...")

    metadata = data.load_metadata(dataset_root, metadata_file)
    logger.debug(f"found metadata: {metadata}")

    patient_directories = data.get_patient_directories(dataset_root)
    patient_ids = data.get_patient_ids(patient_directories)
    logger.debug(f"found patient IDs: {patient_ids}")

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
            dataset_root,
            split,
            patient_id,
            transformed_patient_data,
            patient_metadata,
            buffer,
        )
    logger.info(
        f"patient data preprocessed and saved to: {dataset_root.joinpath('preprocessed')}"
    )
    return


@app.command()
def train_val_test_split(
    dataset_split_path: Annotated[Path, typer.Option()],
    split_ratios: Annotated[str, typer.Option()] = None,
    patient_id_split_json: Annotated[Path, typer.Option()] = None,
):
    if patient_id_split_json is not None:
        with open(patient_id_split_json, "r") as jf:
            patient_id_split = json.load(jf)
    else:
        patient_id_split = None
        split_ratios = split_ratios.split(",")
        split_ratios = [int(split_ratio) for split_ratio in split_ratios]

    data.split_dataset(dataset_split_path, split_ratios, patient_id_split)


def train_single_model(
    model_name: str,
    modality: str,
    target_dir: Path,
    dataset_root: Path,
    split: str,
    mode: str,
    target_size: list,
    num_classes: int,
    batch_size: int,
    shuffle: bool,
    hflip: bool,
    learning_rate: float,
    epochs: int,
    wd: float,
    dropout: float,
    early_stopping_patience: int,
    save_frequency: int,
    scheduler: str,
    run: wandb.run,
):
    logger.info(f"training model: {model_name}")

    use_dct = True if "DCT" in modality else False
    logger.info(f"using DCT: {use_dct}")

    use_gabor = True if "Gabor" in modality else False
    logger.info(f"using Gabor: {use_gabor}")

    use_glcm = True if "GLCM" in modality else False
    logger.info(f"using GLCM: {use_glcm}")

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
        [256, 256] if use_glcm else target_size,
        num_classes,
        use_dct,
        use_gabor,
        use_glcm,
        batch_size,
        shuffle,
        hflip,
    )
    train.train_model(
        model_name,
        modality,
        in_channels,
        current_target_dir,
        train_loader,
        val_loader,
        batch_size,
        learning_rate,
        epochs,
        wd,
        dropout,
        early_stopping_patience,
        [256, 256] if use_glcm else target_size,
        save_frequency,
        scheduler,
        run,
    )
    return


def sweep_trainer():
    run = wandb.init()

    timestamp = wandb.config.timestamp
    model_name = wandb.config.model_name
    modality = wandb.config.modality
    target_size = wandb.config.target_size
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs
    wd = wandb.config.wd
    dropout = wandb.config.dropout
    early_stopping_patience = wandb.config.early_stopping_patience
    save_frequency = wandb.config.save_frequency
    scheduler = wandb.config.scheduler
    dataset_root = Path(wandb.config.dataset_root).resolve()
    split = wandb.config.split
    mode = wandb.config.mode
    num_classes = wandb.config.num_classes
    shuffle = wandb.config.shuffle
    hflip = wandb.config.hflip

    target_str = str(",".join([str(dim) for dim in target_size]))
    target_dir = Path(
        f"sweep/{timestamp}/{target_str}/{learning_rate}/{batch_size}/{wd}/{dropout}/"
    ).resolve()

    logger.info(f"training model: {model_name}")

    use_dct = True if "DCT" in modality else False
    logger.info(f"using DCT: {use_dct}")

    use_gabor = True if "Gabor" in modality else False
    logger.info(f"using Gabor: {use_gabor}")

    use_glcm = True if "GLCM" in modality else False
    logger.info(f"using GLCM: {use_glcm}")

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
        [256, 256] if use_glcm else target_size,
        num_classes,
        use_dct,
        use_gabor,
        use_glcm,
        batch_size,
        shuffle,
        hflip,
    )
    train.train_model(
        model_name,
        modality,
        in_channels,
        current_target_dir,
        train_loader,
        val_loader,
        batch_size,
        learning_rate,
        epochs,
        wd,
        dropout,
        early_stopping_patience,
        [256, 256] if use_glcm else target_size,
        save_frequency,
        scheduler,
        run,
    )

    return


@app.command()
def sweep():
    sweep_config = {
        "method": "grid",
        "metric": {"name": "val_auc", "goal": "maximize"},
        "parameters": {
            "model_name": {"values": ["cnn", "densenet", "swin", "svm"]},
            "modality": {
                "values": [
                    "US",
                    "PA",
                    "PAUS",
                    "DCTUS",
                    "DCTPA",
                    "DCTPAUS",
                    "GaborUS",
                    "GaborPA",
                    "GaborPAUS",
                    "GLCMUS",
                    "GLCMPA",
                    "GLCMPAUS",
                ]
            },
            "batch_size": {"values": [1, 2, 4, 8, 16, 32, 64]},
            "dataset_root": {
                "value": str(Path("dataset/preprocessed").resolve()),
            },
            "dropout": {"values": [0.3, 0.5, 0.7]},
            "early_stopping_patience": {"value": 50},
            "epochs": {"value": 1000},
            "hflip": {"value": False},
            "learning_rate": {"values": [1e-3, 1e-4]},
            "mode": {"value": "train"},
            "num_classes": {"value": 2},
            "save_frequency": {"value": 10},
            "scheduler": {"value": "exponential"},
            "shuffle": {"value": True},
            "split": {"value": "bal"},
            "target_size": {"values": [[150, 390], [256, 256]]},
            "timestamp": {"value": datetime.now().strftime("%Y%m%d%H%M%S")},
            "wd": {"values": [1e-4, 1e-5]},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="paus_2dcs")
    wandb.agent(sweep_id, function=sweep_trainer)
    return


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
    epochs: Annotated[int, typer.Option()] = 1000,
    wd: Annotated[float, typer.Option()] = 1e-4,
    dropout: Annotated[float, typer.Option()] = 0.5,
    early_stopping_patience: Annotated[int, typer.Option()] = 10,
    save_frequency: Annotated[int, typer.Option()] = 10,
    json_path: Annotated[Path, typer.Option()] = None,
    hflip: Annotated[bool, typer.Option()] = False,
    scheduler: Annotated[str, typer.Option()] = "exponential",
    sweep: Annotated[bool, typer.Option()] = False,
):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    mode = "train"

    if not Path.exists(output_dir):
        logger.info(f"creating parent output directory: {output_dir}")
        Path.mkdir(output_dir)
    if output_dir == Path("output").resolve():
        target_dir = output_dir.joinpath(timestamp)
        Path.mkdir(target_dir)
    else:
        target_dir = output_dir

    logger.info(f"created output directory: {target_dir}")

    if json_path is not None:
        shutil.copy(json_path, target_dir)

        with open(json_path, "r") as jf:
            config = json.load(jf)

        model_names = config["model_names"]
        modalities = config["modalities"]
        target_size = config["target_size"]
        batch_size = config["batch_size"]
        learning_rate = config["learning_rate"]
        epochs = config["epochs"]
        wd = config["wd"]
        dropout = config["dropout"]
        early_stopping_patience = config["early_stopping_patience"]
        save_frequency = config["save_frequency"]
        scheduler = config["scheduler"]
    else:
        model_names = model_names.split(",")
        logger.debug(f"model names: {model_names}")

        modalities = modalities.split(",")
        logger.debug(f"modalities: {modalities}")

        target_size = target_size.split(",")
        target_size = [int(size) for size in target_size]
    try:
        for modality in modalities:
            logger.info(f"training for modality: {modality}")
            for model_name in model_names:
                run = wandb.init(
                    project="paus_2dcs",
                    name=f"{timestamp}_{model_name}_{modality}_{split}",
                    config={
                        "model_names": model_names,
                        "modalities": modalities,
                        "target_size": target_size,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "epochs": epochs,
                        "wd": wd,
                        "dropout": dropout,
                        "early_stopping_patience": early_stopping_patience,
                        "save_frequency": save_frequency,
                        "scheduler": scheduler,
                    },
                )
                train_single_model(
                    model_name,
                    modality,
                    target_dir,
                    dataset_root,
                    split,
                    mode,
                    target_size,
                    num_classes,
                    batch_size,
                    shuffle,
                    hflip,
                    learning_rate,
                    epochs,
                    wd,
                    dropout,
                    early_stopping_patience,
                    save_frequency,
                    scheduler,
                    run,
                )
    except Exception as e:
        logger.error(e)
        logger.error("training failed!")
        target_dir.joinpath("failed").touch()
        return
    else:
        logger.info("SUCCESS")
        target_dir.joinpath("completed").touch()
        return


@app.command()
def tester(
    weights_root: Annotated[Path, typer.Option()],
    dataset_root: Annotated[Path, typer.Option()] = Path("dataset/preprocessed"),
    split: Annotated[str, typer.Option()] = "bal",
    target_size: Annotated[str, typer.Option()] = "150,390",
    num_classes: Annotated[int, typer.Option()] = 2,
    device: Annotated[str, typer.Option()] = None,
):
    target_size = target_size.split(",")
    target_size = [int(size) for size in target_size]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"using device: {device}")

    logger.info(f"weights_root: {weights_root.resolve()}")

    weights_paths = natsorted(glob(f"{weights_root}/*/*/best*.pth"))

    if Path.exists(weights_root.joinpath("svm")):
        logger.info("found SVM weights")
        svm_paths = natsorted(glob(f"{weights_root}/svm/*/*.pkl"))

        weights_paths.extend(svm_paths)

    for weights_path in weights_paths:
        logger.info(f"weights_paths: {weights_path}")
        model_name = weights_path.split("/")[-3]
        logger.info(f"model: {model_name}")

        modality = weights_path.split("/")[-2]
        logger.info(f"modality: {modality}")

        use_dct = True if "DCT" in modality else False
        logger.info(f"using DCT: {use_dct}")

        use_gabor = True if "Gabor" in modality else False
        logger.info(f"using Gabor: {use_gabor}")

        use_glcm = True if "GLCM" in modality else False
        logger.info(f"using GLCM: {use_glcm}")

        test_loader = train.get_dataloaders(
            dataset_root,
            split,
            "test",
            modality,
            [256, 256] if use_glcm else target_size,
            num_classes,
            use_dct,
            use_gabor,
            use_glcm,
            batch_size=1,
            shuffle=False,
        )

        train.test_model(
            weights_path,
            model_name,
            modality,
            test_loader,
            device,
            [256, 256] if use_glcm else target_size,
        )


if __name__ == "__main__":
    app()
