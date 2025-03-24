# 3D Image Generator
An interactive application for creating stereoscopic 3D anaglyph images by inserting segmented people into 3D scenes.

## Live Demo
[HuggingFace Link](https://huggingface.co/spaces/dionjin/3D-Image-Composer)

## Usage
Simply upload an image containing a person, select your desired depth level (close, medium, far), and adjust positioning using the sliders. The app automatically segments the person, places them into a stereoscopic scene at the specified depth, and generates an anaglyph image that can be viewed with red-cyan 3D glasses for an immersive effect.

## Customization
Scene Images: Replace scene.jpg with your own side-by-side stereoscopic image
Depth Settings: Modify the disparity values in the disparity_map dictionary
Auto-scaling: Adjust the target_height_ratio parameter to change the relative size of the person

## Example Results
![image](https://github.com/user-attachments/assets/de86f488-0698-4f66-8f89-6a3143ff64b9)
