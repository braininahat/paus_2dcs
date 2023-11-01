from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.svm import SVC
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from asclepius import models
from asclepius.config import logger
from asclepius.data import BreastDataset


def get_dataloaders(
    dataset_root: Path = Path("dataset/preprocessed").resolve(),
    split: str = "bal",
    mode: str = "train",
    modality: str = "PAUS",
    target_size: List[int] = (150, 390),
    num_classes: int = 2,
    use_dct: bool = False,
    use_gabor: bool = False,
    batch_size: int = 32,
    shuffle: bool = True,
):
    dataset_dir = dataset_root.joinpath(split)
    if not Path.exists(dataset_dir):
        raise ValueError(f"dataset directory not found: {dataset_dir}")
    logger.info(f"found dataset directory: {dataset_dir}")

    if mode == "train":
        train_dataset = BreastDataset(
            f"{dataset_dir}/train",
            modality,
            target_size,
            num_classes,
            use_dct,
            use_gabor,
        )
        val_dataset = BreastDataset(
            f"{dataset_dir}/val",
            modality,
            target_size,
            num_classes,
            use_dct,
            use_gabor,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        return train_loader, val_loader
    elif mode == "test":
        test_dataset = BreastDataset(
            f"{dataset_dir}/test",
            modality,
            target_size,
            num_classes,
            use_dct,
            use_gabor,
        )

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return test_loader


def train_and_validate(
    model,
    optimizer,
    scheduler,
    criterion,
    train_loader,
    val_loader,
    epochs,
    output_dir,
    save_frequency,
    early_stopping_patience,
    device,
):
    best_auc = 0.0
    no_improvement_count = 0
    # Training loop
    for epoch in tqdm(range(epochs)):
        total_loss = 0.0

        true_labels = []
        predicted_labels = []

        for images, labels in train_loader:
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predicted_labels.extend(torch.argmax(outputs.detach(), dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

        scheduler.step()

        f1_train = f1_score(true_labels, predicted_labels, average="weighted")
        precision_train = precision_score(
            true_labels, predicted_labels, average="weighted"
        )
        recall_train = recall_score(true_labels, predicted_labels, average="weighted")
        fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
        train_auc = auc(fpr, tpr)

        logger.debug(
            f"Epoch [{epoch + 1}/{epochs}] Train Loss: {(total_loss / len(train_loader)):.4f}\tTrain AUC: {train_auc:.2f}\tTrain F1 Score: {f1_train:.2f}\t\tTrain Precision: {precision_train:.2f}\t\t\tTrain Recall: {recall_train:.2f}"
        )

        # Validation

        true_labels = []
        predicted_labels = []

        valid_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.cuda()
                labels = labels.cuda()

                outputs = model(images)

                valid_loss += criterion(outputs, labels)

                predicted_labels.extend(
                    torch.argmax(outputs.detach(), dim=1).cpu().numpy()
                )
                true_labels.extend(labels.cpu().numpy())

        # Calculate F1 score, precision, and recall

        f1_val = f1_score(true_labels, predicted_labels, average="weighted")
        precision_val = precision_score(
            true_labels, predicted_labels, average="weighted"
        )
        recall_val = recall_score(true_labels, predicted_labels, average="weighted")
        fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
        val_auc = auc(fpr, tpr)

        if val_auc > best_auc:
            no_improvement_count = 0
            best_auc = val_auc
            torch.save(model.state_dict(), f"{output_dir}/best_auc.pth")
            logger.info("Saved the best model.")
        else:
            no_improvement_count += 1

        if (epoch + 1) % save_frequency == 0:
            torch.save(model.state_dict(), f"{output_dir}/epoch_{epoch + 1}.pth")
        logger.info(
            f"Epoch [{epoch + 1}/{epochs}]\t\t\t Validation Loss: {valid_loss/len(val_loader):.4f}\tValidation AUC: {val_auc:.2f}\tValidation F1 Score: {f1_val:.2f}\tValidation Precision: {precision_val:.2f}\tValidation Recall: {recall_val:.2f}"
        )

        if no_improvement_count >= early_stopping_patience:
            logger.info(
                f"Early stopping at epoch {epoch + 1} due to no improvement in validation AUC."
            )
            break

        model.train()


def train_model(
    model_name: str,
    modality: str,
    in_channels: int,
    output_dir: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    batch_size: int = 16,
    learning_rate: float = 1e-6,
    epochs: int = 1000,
    wd: float = 0,
    dropout: float = 0.5,
    early_stopping_patience: int = 50,
    target_size: List[int] = [150, 390],
    save_frequency: int = 10,
):
    if model_name == "svm":
        model = SVC(kernel="linear", probability=True)

        for images, labels in train_loader:
            images = images.view(images.shape[0], -1)
            model.fit(images, labels)

        true_labels = []
        predicted_labels = []

        for images, labels in val_loader:
            images = images.view(images.shape[0], -1)
            predicted_labels.extend(model.predict(images))
            true_labels.extend(labels.numpy())

        f1_val = f1_score(true_labels, predicted_labels, average="weighted")
        precision_val = precision_score(
            true_labels, predicted_labels, average="weighted"
        )
        recall_val = recall_score(true_labels, predicted_labels, average="weighted")
        fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
        val_auc = auc(fpr, tpr)

        logger.info(
            f"Validation AUC: {val_auc:.2f}\tValidation F1 Score: {f1_val:.2f}\tValidation Precision: {precision_val:.2f}\tValidation Recall: {recall_val:.2f}"
        )

        # Save model
        joblib.dump(model, f"{output_dir}/svm.pkl")

    else:
        if model_name == "cnn":
            model = models.BreastCNN(in_channels=in_channels, image_size=target_size)
        elif model_name == "swin":
            model = models.custom_swin_t(in_channels=in_channels)
        elif model_name == "vit":
            model = models.custom_vit_b_16(in_channels=in_channels)
        elif model_name == "densenet":
            model = models.custom_densenet(in_channels=in_channels)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=wd
        )
        scheduler = CosineAnnealingLR(optimizer, len(train_loader) / batch_size)
        criterion = nn.CrossEntropyLoss()

        train_and_validate(
            model,
            optimizer,
            scheduler,
            criterion,
            train_loader,
            val_loader,
            epochs,
            output_dir,
            save_frequency,
            early_stopping_patience,
            device,
        )

    logger.info("Training finished!")


def test_model(
    weights_path: Path,
    model_name: str,
    modality: str,
    test_loader: DataLoader,
    device: str = "cuda:0",
    target_size: List[int] = [150, 390],
):
    output_path = Path(weights_path).parent.joinpath("results")
    output_path.mkdir(parents=True, exist_ok=True)

    if model_name == "svm":
        model = joblib.load(weights_path)

        true_labels = []
        predicted_labels = []
        predicted_probs = []

        for images, labels in test_loader:
            images = images.view(images.shape[0], -1)
            predicted_probs.extend(model.predict_proba(images))
            predicted_labels.extend(model.predict(images))
            true_labels.extend(labels.numpy())
    else:
        if "Dual" in modality:
            in_channels = 4
        elif "PAUS" in modality:
            in_channels = 2
        else:
            in_channels = 1

        if model_name == "cnn":
            model = models.BreastCNN(in_channels=in_channels, image_size=target_size)
        elif model_name == "swin":
            model = models.custom_swin_t(in_channels=in_channels)
        elif model_name == "vit":
            model = models.custom_vit_b_16(in_channels=in_channels)
        elif model_name == "densenet":
            model = models.custom_densenet(in_channels=in_channels)

        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
        except FileNotFoundError as e:
            logger.error(f"{weights_path} not found")
            return
        model.to(device)
        model.eval()

        true_labels = []
        predicted_labels = []
        predicted_probs = []

        with torch.no_grad():
            for image, label in test_loader:
                image = image.to(device)
                label = label.to(device)

                output = model(image)

                predicted_probs.extend(output.detach().cpu().numpy())
                predicted_labels.extend(
                    torch.argmax(output.detach(), dim=1).cpu().numpy()
                )  # Threshold at 0.5 for binary classification

                true_labels.extend(label.cpu().numpy())

    # Calculate F1 score, precision, and recall

    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    precision = precision_score(true_labels, predicted_labels, average="weighted")
    recall = recall_score(true_labels, predicted_labels, average="weighted")

    logger.info(f"Test results for {model_name} - {modality}:")
    logger.info(f"F1 Score: {f1:.2f}\tPrecision: {precision:.2f}\tRecall: {recall:.2f}")

    pred_gt_pairs = np.vstack([np.squeeze(np.array(predicted_labels).T), true_labels]).T
    preds = np.squeeze(np.array(predicted_labels))
    gts = np.squeeze(np.array(true_labels))

    hmap = sns.heatmap(pred_gt_pairs, cbar=False)
    fig = hmap.get_figure()

    fig.savefig(output_path.joinpath("best_auc_heatmap.png"))
    plt.close(fig)

    # Calculate ROC curve and AUC for validation set
    fpr, tpr, thresholds = roc_curve(
        gts, np.array(predicted_probs)[:, 1], drop_intermediate=False
    )
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = {:.2f})".format(roc_auc),
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(output_path.joinpath("best_auc_roc.png"))
    plt.close()

    np.savez(
        output_path.joinpath("metrics.npz"),
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        roc_auc=roc_auc,
    )

    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "roc_auc": roc_auc}
