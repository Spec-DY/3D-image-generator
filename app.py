import os
import numpy as np
import cv2
import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101
from pathlib import Path
import tempfile

SCENE_IMAGE_PATH = "scene.jpg"


def load_segmentation_model():
    model = deeplabv3_resnet101(weights='DEFAULT')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def segment_person(img_data, model):
    """
    Extracts a person from the input image using a semantic segmentation model and returns an RGBA image 
    with a transparent background where the person is isolated.

    The function writes the input image to a temporary file, loads and preprocesses it, and then performs 
    segmentation using the provided model. It applies morphological operations to clean the segmentation mask, 
    and finally combines the original image with the mask to produce an RGBA image.

    Parameters:
        img_data (numpy.ndarray): The input image data in RGB format.
        model (torch.nn.Module): A pre-trained segmentation model (e.g., DeepLabV3) in evaluation mode.

    Returns:
        numpy.ndarray: An RGBA image with the person segmented out (alpha channel representing the person mask).
    """
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_path = temp_file.name
        cv2.imwrite(temp_path, cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))

    try:

        input_image = Image.open(temp_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        # Perform segmentation
        with torch.no_grad():
            output = model(input_batch)['out'][0]

        output_predictions = output.argmax(0).byte().cpu().numpy()
        person_mask = (output_predictions == 15).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)

        original_image = cv2.imread(temp_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        h, w = original_image.shape[:2]
        rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_image[:, :, :3] = original_image
        rgba_image[:, :, 3] = person_mask

        return rgba_image

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def auto_scale_person(person_rgba, scene_img, target_height_ratio=0.7):
    """
    Automatically scale the person image to fit the scene

    Args:
        person_rgba: Person image with transparency channel
        scene_img: Scene image
        target_height_ratio: Proportion of scene height that the person should occupy

    Returns:
        Scaled person image and scale ratio
    """

    scene_h, scene_w = scene_img.shape[:2]
    person_h, person_w = person_rgba.shape[:2]
    target_height = int(scene_h * target_height_ratio)
    scale_ratio = target_height / person_h
    new_width = int(person_w * scale_ratio)
    scaled_person = cv2.resize(person_rgba, (new_width, target_height))

    return scaled_person, scale_ratio


def split_stereo_image(img_path):
    """
    Extract left and right stereo images from a side-by-side stereo image
    """

    stereo_img = cv2.imread(img_path)
    if stereo_img is None:
        raise FileNotFoundError(f"Scene image not found at {img_path}")

    stereo_rgb = cv2.cvtColor(stereo_img, cv2.COLOR_BGR2RGB)

    # Get dimensions
    h, w = stereo_rgb.shape[:2]

    # Split into left and right
    # here we assume image is always side-by-side stereo with equal width for left and right
    mid_point = w // 2
    left_img = stereo_rgb[:, :mid_point]
    right_img = stereo_rgb[:, mid_point:]

    return left_img, right_img


def insert_into_stereo_image(person_rgba, stereo_left, stereo_right, depth_level, x_position, y_position, scale_factor=1.0):
    """
    Inserts a segmented person image into left and right stereo images with a depth-dependent horizontal offset,
    and creates an anaglyph image.

    Parameters:
        person_rgba (numpy.ndarray): The RGBA image of the segmented person.
        stereo_left (numpy.ndarray): The left background stereo image.
        stereo_right (numpy.ndarray): The right background stereo image.
        depth_level (str): Depth setting ("close", "medium", "far") that determines the disparity.
        x_position (int or float): Horizontal position for overlaying the person.
        y_position (int or float): Vertical position for overlaying the person.
        scale_factor (float, optional): Scaling factor to adjust the size of the person image (default is 1.0).

    Returns:
        tuple: A tuple containing:
            - stereo_left_modified (numpy.ndarray): Left stereo image with the person overlaid.
            - stereo_right_modified (numpy.ndarray): Right stereo image with the person overlaid.
            - anaglyph (numpy.ndarray): The combined red-cyan anaglyph image.
    """
    # disparity for each depth level
    disparity_map = {
        "close": 30,
        "medium": 15,
        "far": 5
    }
    disparity = disparity_map[depth_level]

    stereo_left_modified = stereo_left.copy()
    stereo_right_modified = stereo_right.copy()

    if scale_factor != 1.0:
        h, w = person_rgba.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        person_resized = cv2.resize(person_rgba, (new_w, new_h))
    else:
        person_resized = person_rgba

    h_stereo, w_stereo = stereo_left_modified.shape[:2]

    x_position = min(max(0, x_position), w_stereo - 1)
    y_position = min(max(0, y_position), h_stereo - 1)

    person_h, person_w = person_resized.shape[:2]
    x_position = min(x_position, w_stereo - person_w)
    y_position = min(y_position, h_stereo - person_h)

    left_x = max(0, int(x_position - disparity/2))
    right_x = max(0, int(x_position + disparity/2))

    left_x = min(left_x, w_stereo - person_w)
    right_x = min(right_x, w_stereo - person_w)

    # Get the actual regions of interest with proper bounds checking
    try:

        roi_left = stereo_left_modified[y_position:y_position +
                                        person_h, left_x:left_x+person_w]
        # Use alpha blending
        for c in range(3):
            alpha_mask = person_resized[:, :, 3] / 255.0
            roi_left[:, :, c] = roi_left[:, :, c] * \
                (1 - alpha_mask) + person_resized[:, :, c] * alpha_mask

        stereo_left_modified[y_position:y_position +
                             person_h, left_x:left_x+person_w] = roi_left

        # Right image ROI
        roi_right = stereo_right_modified[y_position:y_position +
                                          person_h, right_x:right_x+person_w]
        # Use alpha blending
        for c in range(3):
            roi_right[:, :, c] = roi_right[:, :, c] * \
                (1 - alpha_mask) + person_resized[:, :, c] * alpha_mask

        stereo_right_modified[y_position:y_position +
                              person_h, right_x:right_x+person_w] = roi_right
    except Exception as e:
        print(f"Error during image blending: {e}")

        if 'roi_left' in locals() and 'person_resized' in locals():
            print(
                f"Left ROI shape: {roi_left.shape}, Person shape: {person_resized.shape}")
            if hasattr(person_resized, 'shape') and len(person_resized.shape) > 2:
                print(f"Alpha shape: {person_resized[:, :, 3].shape}")

        return stereo_left, stereo_right, create_anaglyph(stereo_left, stereo_right)

    # Create anaglyph image
    anaglyph = create_anaglyph(stereo_left_modified, stereo_right_modified)

    return stereo_left_modified, stereo_right_modified, anaglyph


def create_anaglyph(left_img, right_img):
    """
    Convert stereo image pair to anaglyph format for red-cyan glasses.
    """
    # Extract color channels
    left_red = left_img[:, :, 0]
    right_green = right_img[:, :, 1]
    right_blue = right_img[:, :, 2]

    anaglyph = np.zeros_like(left_img)
    anaglyph[:, :, 0] = left_red  # Red channel from left image
    anaglyph[:, :, 1] = right_green  # Green channel from right image
    anaglyph[:, :, 2] = right_blue  # Blue channel from right image

    return anaglyph


def process_image(input_image, depth_level, x_position, y_position, scale_factor, auto_scale=True):
    """
    Segments a person from the input image and overlays them onto a stereo background to create an anaglyph image.

    Parameters:
        input_image (PIL.Image): The image containing a person.
        depth_level (str): Depth level ("close", "medium", "far") that determines the stereo disparity.
        x_position (int/float): Horizontal offset for positioning.
        y_position (int/float): Vertical offset for positioning.
        scale_factor (float): Additional scaling factor for the segmented person.
        auto_scale (bool): If True, automatically scales the person to fit the background (default True).

    Returns:
        tuple: (segmented_person, anaglyph)
    """
    try:
        model = load_segmentation_model()
        segmented_person = segment_person(np.array(input_image), model)
    except Exception as e:
        print(f"Error during segmentation: {e}")

        original_rgb = np.array(input_image)
        h, w = original_rgb.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        center_h, center_w = h//2, w//2
        mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 255

        segmented_person = np.zeros((h, w, 4), dtype=np.uint8)
        segmented_person[:, :, :3] = original_rgb
        segmented_person[:, :, 3] = mask

    # Split stereo image into left and right
    try:
        stereo_left, stereo_right = split_stereo_image(SCENE_IMAGE_PATH)
    except FileNotFoundError:
        print(
            f"Scene image not found at {SCENE_IMAGE_PATH}. Using a default background.")
        # If scene.jpg not found, use a default color as background
        h, w = segmented_person.shape[:2]
        stereo_left = np.ones((h, w, 3), dtype=np.uint8) * \
            128  # Gray background
        stereo_right = stereo_left.copy()

    if auto_scale:
        segmented_person, auto_scale_factor = auto_scale_person(
            segmented_person, stereo_left, 0.7)
        print(f"Auto-scaled person with ratio: {auto_scale_factor:.2f}")
        # Apply additional scaling if specified by user
        if scale_factor != 1.0:
            h, w = segmented_person.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            segmented_person = cv2.resize(segmented_person, (new_w, new_h))
    else:

        h, w = segmented_person.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        segmented_person = cv2.resize(segmented_person, (new_w, new_h))

    try:
        _, _, anaglyph = insert_into_stereo_image(
            segmented_person, stereo_left, stereo_right, depth_level,
            int(x_position), int(y_position), 1.0
        )
    except Exception as e:
        print(f"Error during stereo insertion: {e}")
        anaglyph = create_anaglyph(stereo_left, stereo_right)

    return segmented_person, anaglyph


def create_gradio_interface():
    with gr.Blocks(title="3D Image Composer") as app:
        gr.Markdown("# 3D Image Composer")
        gr.Markdown(
            "Upload an image with a person to create an anaglyph 3D image.")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image (with person)")
                depth_choice = gr.Radio(
                    choices=["close", "medium", "far"],
                    label="Depth Level",
                    value="medium"
                )
                x_position = gr.Slider(
                    minimum=0, maximum=1000, value=200, step=10, label="Horizontal Position")
                y_position = gr.Slider(
                    minimum=0, maximum=1000, value=1000, step=10, label="Vertical Position")
                auto_scale_checkbox = gr.Checkbox(
                    label="Auto-scale person to fit scene", value=True)
                scale_factor = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                                         label="Additional scale factor")

                process_btn = gr.Button("Process Image")

            with gr.Column():
                segmented_output = gr.Image(label="Segmented Person")
                anaglyph_output = gr.Image(
                    label="Anaglyph Image (Use red-cyan glasses)")

        gr.Markdown("### Instructions")
        gr.Markdown("""
        1. Upload an image containing a person.
        2. Choose the depth level (close, medium, far).
        3. Adjust the position of the person in the scene.
        4. Choose whether to auto-scale the person to fit the scene.
        5. Click 'Process Image' to generate the 3D anaglyph.
        6. View the final anaglyph image with red-cyan 3D glasses.
        """)

        process_btn.click(
            fn=process_image,
            inputs=[input_image, depth_choice, x_position, y_position,
                    scale_factor, auto_scale_checkbox],
            outputs=[segmented_output, anaglyph_output]
        )

    return app


if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch()
