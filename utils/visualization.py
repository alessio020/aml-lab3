import matplotlib.pyplot as plt
import numpy as np


# Function to denormalize image for visualization
def denormalize(image):
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image


# Visualize one example for each class for 10 classes
def visualize_samples(dataloader, dataset, num_classes_to_show=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    classes_sampled = []
    found_classes = 0

    for i, (inputs, classes) in enumerate(dataloader):
        for image, label in zip(inputs, classes):
            class_id = label.item()

            if class_id not in classes_sampled:
                classes_sampled.append(class_id)

                row = found_classes // 5
                col = found_classes % 5

                axes[row, col].imshow(denormalize(image))
                axes[row, col].set_title(dataset.classes[class_id])
                axes[row, col].axis('off')

                found_classes += 1

            if found_classes == num_classes_to_show:
                break

        if found_classes == num_classes_to_show:
            break

    plt.tight_layout()
    plt.show()

