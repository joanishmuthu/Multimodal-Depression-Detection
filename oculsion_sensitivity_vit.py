import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from transformers import ViTImageProcessor, ViTForImageClassification


def predict_and_visualize(image_path, model_path):
    # Load the trained model
    model = ViTForImageClassification.from_pretrained(model_path)
    # Load the image processor
    processor = ViTImageProcessor.from_pretrained(model_path)

    # Define transformations for preprocessing the image
    image_size = processor.size["height"]
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        normalize
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        outputs = model(input_tensor)

    # Get predicted probabilities and predicted label
    predicted_probabilities = torch.softmax(outputs.logits, dim=-1)[0].numpy()
    predicted_label_id = predicted_probabilities.argmax()
    predicted_label = model.config.id2label[predicted_label_id]

    # Perform Occlusion Sensitivity
    occlusion_sensitivity = compute_occlusion_sensitivity(model, input_tensor)

    # Plot original image and occlusion sensitivity map
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    ax[1].imshow(occlusion_sensitivity, cmap='hot', interpolation='nearest')
    ax[1].set_title("Occlusion Sensitivity")
    ax[1].axis('off')

    plt.show()

    return predicted_label, predicted_probabilities


def compute_occlusion_sensitivity(model, input_tensor, patch_size=32):
    # Get model's prediction without occlusion
    with torch.no_grad():
        output = model(input_tensor)
        original_prediction = torch.softmax(output.logits, dim=-1)[0, :].numpy()

    # Get image dimensions
    batch_size, channels, height, width = input_tensor.shape

    # Initialize an empty array to store occlusion sensitivity scores
    occlusion_sensitivity = np.zeros((height, width))

    # Iterate through the image with a sliding window of patch_size
    for h in range(0, height, patch_size):
        for w in range(0, width, patch_size):
            # Create a copy of the input tensor with occluded patch
            occluded_input = input_tensor.clone()
            occluded_input[:, :, h:h + patch_size, w:w + patch_size] = 0

            # Get model's prediction with occlusion
            with torch.no_grad():
                output = model(occluded_input)
                occluded_prediction = torch.softmax(output.logits, dim=-1)[0, :].numpy()

            # Compute the difference in predictions
            difference = original_prediction - occluded_prediction
            occlusion_sensitivity[h:h + patch_size, w:w + patch_size] = np.linalg.norm(difference, axis=-1)

    return occlusion_sensitivity

#
# # # Example usage:
# image_path = "Expw-F/happy/4angry_actor_13.jpg"
# model_path = "new_model/"
# predicted_label, predicted_probabilities = predict_and_visualize(image_path, model_path)
#
# print("Predicted Label:", predicted_label)
# print("Predicted Probabilities:", predicted_probabilities)
