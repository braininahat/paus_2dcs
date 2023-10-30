from glob import glob
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from natsort import natsorted
from sklearn.metrics import auc, f1_score, precision_score, recall_score, roc_curve
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

    elif split == "test":
        test_dataset = BreastDataset(
            f"{dataset_dir}/test",
            modality,
            target_size,
            num_classes,
            use_dct,
            use_gabor,
        )

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

        return test_loader


def train_model(
    model_name: str,
    in_channels: int,
    output_dir: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    batch_size: int = 16,
    learning_rate: float = 1e-6,
    num_epochs: int = 1000,
    wd: float = 0,
    dropout: float = 0.5,
    early_stopping_patience: int = 50,
    target_size: List[int] = [150, 390],
):
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, len(train_loader) / batch_size)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    no_improvement_count = 0

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0

        true_labels = []
        predicted_labels = []

        for images, labels in train_loader:
            optimizer.zero_grad()

            images = images.cuda()
            labels = labels.cuda()

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

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {(total_loss / len(train_loader)):.4f}\tTrain AUC: {train_auc:.2f}\tTrain F1 Score: {f1_train:.2f}\t\tTrain Precision: {precision_train:.2f}\t\t\tTrain Recall: {recall_train:.2f}"
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

        torch.save(model.state_dict(), f"{output_dir}/epoch_{epoch + 1}.pth")
        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}]\t\t\t Validation Loss: {valid_loss/len(val_loader):.4f}\tValidation AUC: {val_auc:.2f}\tValidation F1 Score: {f1_val:.2f}\tValidation Precision: {precision_val:.2f}\tValidation Recall: {recall_val:.2f}"
        )

        if no_improvement_count >= early_stopping_patience:
            logger.info(
                f"Early stopping at epoch {epoch + 1} due to no improvement in validation AUC."
            )
            break

        model.train()

    logger.info("Training finished!")


# def test_model(
#     test_loaders: List[Dict[str, DataLoader]],
#     model_name: str,
# ):
#     all_fpr = []
#     all_tpr = []
#     all_roc_auc = []
#     all_thresholds = []

#     plot_labels = {
#         "US": "Image(US)+CNN",
#         "PA": "Image(PA)+CNN",
#         "PAUS": "Image(PAUS)+CNN",
#         "DCTUS": "DCT(US)+CNN",
#         "DCTPA": "DCT(PA)+CNN",
#         "DCTPAUS": "DCT(PAUS)+CNN",
#         "GaborUS": "Gabor(US)+CNN",
#         "GaborPA": "Gabor(PA)+CNN",
#         "GaborPAUS": "Gabor(PAUS)+CNN",
#         "DualPAUS": "[Image(PAUS)+DCT(PAUS)]+CNN",
#     }

#         logger.info(f"Dataset Split: {dataset_split}, Modality: {modality}")

#         if "Dual" in modality:
#             in_channels = 4
#         elif "PAUS" in modality:
#             in_channels = 2
#         else:
#             in_channels = 1

#         if model_name == "cnn":
#             model = models.BreastCNN(in_channels=in_channels)
#         elif model_name == "swin":
#             model = models.custom_swin_t(in_channels=in_channels)
#         elif model_name == "vit":
#             model = models.custom_vit_b_16(in_channels=in_channels)

#         model.cuda()

#         try:
#             model.load_state_dict(
#                 torch.load(f"{run_prefix}/{modality}/{dataset_split}/best_auc.pth")
#             )
#         except FileNotFoundError as e:
#             logger.error(
#                 f"No best_auc weights found for {modality} {dataset_split}. Skipping..."
#             )
#             continue

#         model.eval()

#         true_labels = []
#         predicted_labels = []
#         predicted_probs = []

#         with torch.no_grad():
#             for image, label in test_loader:
#                 image = image.cuda()
#                 label = label.cuda()
#                 output = model(image)
#                 predicted_probs.extend(output.detach().cpu().numpy())
#                 predicted_labels.extend(
#                     torch.argmax(output.detach(), dim=1).cpu().numpy()
#                 )  # Threshold at 0.5 for binary classification
#                 true_labels.extend(label.cpu().numpy())

#         # Calculate F1 score, precision, and recall

#         f1 = f1_score(true_labels, predicted_labels, average="weighted")
#         precision = precision_score(true_labels, predicted_labels, average="weighted")
#         recall = recall_score(true_labels, predicted_labels, average="weighted")

#         pred_gt_pairs = np.vstack(
#             [np.squeeze(np.array(predicted_labels).T), true_labels]
#         ).T
#         preds = np.squeeze(np.array(predicted_labels))
#         gts = np.squeeze(np.array(true_labels))
#         vals, counts = np.unique(np.equal(preds, gts), return_counts=True)

#         hmap = sns.heatmap(pred_gt_pairs, cbar=False)
#         fig = hmap.get_figure()
#         if not os.path.exists(f"{run_prefix}/{modality}/{dataset_split}/results/"):
#             os.makedirs(f"{run_prefix}/{modality}/{dataset_split}/results/")
#         fig.savefig(f"{run_prefix}/{modality}/{dataset_split}/results/best_auc.png")
#         plt.close(fig)

#         # Calculate ROC curve and AUC for validation set
#         fpr, tpr, thresholds = roc_curve(
#             gts, np.array(predicted_probs)[:, 1], drop_intermediate=False
#         )
#         roc_auc = auc(fpr, tpr)

#         # Append the fpr, tpr, and roc_auc to the respective lists
#         all_fpr.append(fpr)
#         all_tpr.append(tpr)
#         all_thresholds.append(thresholds)
#         all_roc_auc.append(roc_auc)

#         # Plot ROC curve
#         plt.figure(figsize=(8, 6))
#         plt.plot(
#             fpr,
#             tpr,
#             color="darkorange",
#             lw=2,
#             label="ROC curve (area = {:.2f})".format(roc_auc),
#         )
#         plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel("False Positive Rate")
#         plt.ylabel("True Positive Rate")
#         plt.title("Receiver Operating Characteristic")
#         plt.legend(loc="lower right")
#         plt.savefig(f"{run_prefix}/{modality}/{dataset_split}/results/best_auc_roc.png")
#         plt.close()

#     # Plot all ROC curves in one figure
#     plt.figure(figsize=(8, 6))
#     for i in range(len(test_loaders)):
#         plt.plot(
#             all_fpr[i],
#             all_tpr[i],
#             lw=2,
#             label=f"{plot_labels[test_loaders[i][1]]} (area = {all_roc_auc[i]:.2f})",
#             linestyle="solid" if "DCT" in test_loaders[i][1] else "dashed",
#         )

#     # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("Receiver Operating Characteristic")
#     plt.legend(loc="lower right")
#     plt.savefig("all_roc_curves.png")  # Save the combined ROC curves
#     plt.show()  # Display the combined ROC curves
