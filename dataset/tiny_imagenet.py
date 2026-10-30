import os
import shutil
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


def prepare_val_folder(root_dir='tiny-imagenet/tiny-imagenet-200'):
    """Adjust the format of the val split of the dataset to be used with ImageFolder."""
    val_images_dir = os.path.join(root_dir, 'val', 'images')
    val_annotations_file = os.path.join(root_dir, 'val', 'val_annotations.txt')

    # Run only if val/images still exists (not yet reorganized)
    if os.path.exists(val_images_dir) and os.path.exists(val_annotations_file):
        with open(val_annotations_file) as f:
            for line in f:
                fn, cls, *_ = line.split('\t')
                cls_dir = os.path.join(root_dir, 'val', cls)
                os.makedirs(cls_dir, exist_ok=True)
                shutil.copyfile(
                    os.path.join(val_images_dir, fn),
                    os.path.join(cls_dir, fn)
                )
        shutil.rmtree(val_images_dir)


def get_dataloaders(root_dir='tiny-imagenet/tiny-imagenet-200', batch_size=32, num_workers=2):
    """Create and return train and validation DataLoaders for Tiny-ImageNet."""
    # Ensure validation folder is properly formatted for ImageFolder
    prepare_val_folder(root_dir)

    transform = T.Compose([
        T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')

    tiny_imagenet_dataset_train = ImageFolder(root=train_dir, transform=transform)
    tiny_imagenet_dataset_val = ImageFolder(root=val_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        tiny_imagenet_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        tiny_imagenet_dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, tiny_imagenet_dataset_train, tiny_imagenet_dataset_val

