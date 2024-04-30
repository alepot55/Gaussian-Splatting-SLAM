# Gaussian Splatting for SLAM

This project aims is a dense Simultaneous Localization and Mapping (SLAM) system that aims to leverage Gaussian splatting to achieve photo-realistic reconstructions. For the last-year excellence program at Sapienza University of Rome, the project focuses on creating a robust model for SLAM using Gaussian representations. SLAM is a critical component in robotics and computer vision, enabling robots and autonomous systems to navigate and map their environment. 

The method builds upon the foundation provided by NerfStudio, a powerful library for 3D reconstruction. Specifically, Gaussian-SLAM utilizes NerfStudio’s Splatfacto method as a starting point. On top of that, a tracking module is implemented to to enhance system robustness against camera motion and occlusions. As a result, unlike many SLAM models that rely on depth information, the model can operates solely on RGB images. Lastly, to maintain a diverse set of images, a covisibility control mechanism aims to select a subset of diverse images to balance the need for coverage with minimizing redundancy.

In the image below it is shown the SLAM system oerview, with the implemented components highlighted:

<p align="center">
    <img src="https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM/assets/61759069/1592bd44-b794-460f-b754-501145c51102" width="100%">
</p>

Professor: _Thomas Alessandro Ciarfuglia_

## Getting Started

1. **Install NerfStudio**:
   - Follow the guidelines provided [here](https://docs.nerf.studio/quickstart/installation.html).

2. **Clone the Repository**:
   - Clone this [repository](https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM) to your local system.

3. **Install this Repository as a Python Package**:
   - Change your directory to the cloned repository folder.
   - Initiate the installation of the package in editable mode by executing the command:
     ```
     python -m pip install -e .
     ```
   - Run `ns-install-cli` to establish the required command-line interface.
   - Verify the successful installation by executing:
     ```
     ns-train -h
     ```
     The `my_method` method should appear in the list of subcommands.

4. **Training the Model**:
   - Launch training using the command:
     ```
     ns-train my_method --data <data_folder>
     ```
     Replace `<data_folder>` with the path to your data folder.
   - You can use either [existing datasets](https://docs.nerf.studio/quickstart/existing_dataset.html) or [create your custom dataset](https://docs.nerf.studio/quickstart/custom_dataset.html).
