# Fine-tuning-a-Vision-Transformer-model-using-LoRA-on-the-PlantVillage-dataset
Plant Disease Classification with Vision Transformer (ViT) and LoRA

This project demonstrates how to fine-tune a pre-trained Vision Transformer (ViT) model for plant disease classification using Low-Rank Adaptation (LoRA).
Project Overview

The goal of this project is to classify plant diseases from images. We leverage a pre-trained ViT model from the Hugging Face transformers library and fine-tune it on a custom dataset using LoRA, a parameter-efficient fine-tuning technique.
Dataset

The dataset used is the PlantVillage dataset, which contains images of various plant leaves, categorized by plant type and disease status. The dataset is sourced from Kaggle (emmarex/plantdisease).
Setup and Installation

To run this notebook, you'll need to install the following Python packages:

!pip install transformers torchvision pillow kaggle peft datasets accelerate

Data Preparation

    Download and Unzip Data: The PlantVillage dataset is downloaded from Kaggle and unzipped into the /content/PlantVillage directory.

    !kaggle datasets download -d emmarex/plantdisease
    !unzip /content/plantdisease.zip

    Define Transformations: Images are resized to 224x224 pixels, converted to PyTorch tensors, and normalized using ImageNet standards.

    from torchvision import transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    Load Dataset: The ImageFolder class from torchvision.datasets is used to load images from the directory structure, automatically assigning labels based on folder names.

    from torchvision import datasets
    data_dir = '/content/PlantVillage'
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    class_names = full_dataset.classes

    Split Data: The dataset is split into training (90%) and validation (10%) sets.

    from torch.utils.data import random_split, DataLoader
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)

Model Fine-tuning with LoRA

    Load Pre-trained ViT Model: A vit-tiny-patch16-224 model is loaded from Hugging Face and reconfigured with the correct number of output classes for our dataset.

    from transformers import AutoImageProcessor, ViTForImageClassification
    model_name = "WinKawaks/vit-tiny-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_name)
    NUM_CLASSES = len(class_names)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True,
        num_labels=NUM_CLASSES
    )

    Configure LoRA: LoRA configuration is applied to the model, targeting the 'query' and 'value' attention modules. This significantly reduces the number of trainable parameters.

    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    Training: The LoRA-adapted model is trained using the AdamW optimizer with a learning rate of 5e-5. Training involves iterating through epochs, calculating loss, performing backpropagation, and updating weights. The model with the best validation loss is saved.
