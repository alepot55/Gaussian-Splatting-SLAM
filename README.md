# Gaussian Splatting for SLAM: Photo-realistic 3D Mapping from RGB Video

## Overview

This project introduces a novel approach to **dense Simultaneous Localization and Mapping (SLAM)**, achieving **photo-realistic 3D reconstructions** using **3D Gaussian Splatting**. Developed as part of the excellence program at Sapienza University of Rome, this research explores the exciting potential of Gaussian representations within SLAM systems. SLAM is a fundamental technology in robotics and computer vision, enabling autonomous robots and systems to navigate and map their surroundings in real-time. Our work pushes the boundaries of visual SLAM by creating highly detailed and visually compelling 3D models directly from standard RGB video input.

**Why is this important?** Traditional SLAM systems often struggle with creating dense, visually appealing maps, or rely on depth sensors (RGB-D). Our method overcomes these limitations by:

*   **Generating high-fidelity 3D models:**  Using Gaussian Splatting, we achieve reconstructions that are significantly more detailed and photo-realistic compared to traditional mesh or point cloud-based SLAM.
*   **Operating with RGB-only video:**  Eliminating the need for depth sensors makes the system more versatile and robust, as it can work with standard cameras and is less susceptible to issues like sensor noise, range limitations, and lighting variations that affect depth sensors. This is crucial for real-world applications where depth information might not be readily available or reliable.
*   **Improving robustness in challenging conditions:** The system is designed to be more robust against common SLAM challenges such as fast camera motion and occlusions (objects temporarily blocking the view), which are typical in dynamic environments.

## Key Features: Building a Robust and Photo-realistic SLAM System

This project builds upon the foundation of NerfStudio and introduces several key innovations to create a robust and visually impressive RGB-D SLAM system:

*   **NerfStudio Integration:**  Leveraging the powerful and modular NerfStudio library provides a solid base for 3D reconstruction and rendering. NerfStudio's optimized infrastructure allows for efficient training and high-quality results.
*   **Splatfacto Adaptation for SLAM:** We adapted NerfStudio's Splatfacto method, originally designed for offline scene reconstruction, to work in a real-time SLAM setting. This involved modifying the core algorithms to handle continuous video input and incremental map building, which is essential for SLAM.
*   **RGB-Only Operation:**  A significant advancement is the system's ability to function solely with RGB images. This is achieved through innovative modifications to the Gaussian Splatting and tracking algorithms, allowing the system to infer 3D structure directly from 2D color information. This drastically expands the applicability of the system.
*   **Photometric Error-Based Tracking Module:** To enhance system robustness, especially during camera motion, we implemented a tracking module based on photometric error. This module optimizes the camera pose by minimizing the difference between rendered views and observed images, leading to more accurate camera tracking and a more stable SLAM system.
*   **Covisibility Control for Efficient Image Selection:**  Efficiently selecting the most informative images for map updates is crucial in SLAM. We incorporated a covisibility control mechanism that intelligently chooses a diverse set of images that maximize the information gain for the 3D map. This improves the quality of the reconstruction and reduces computational redundancy.
*   **Photo-realistic Reconstructions:** The system's core strength lies in its ability to generate highly detailed and photo-realistic 3D scene representations. The use of Gaussian Splatting allows for capturing intricate details and view-dependent effects, resulting in visually stunning and immersive 3D models.

## System Architecture: Component Breakdown

The diagram below illustrates the key components of our Gaussian Splatting SLAM system and their interactions:

![SLAM System Overview](https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM/assets/61759069/1592bd44-b794-460f-b754-501145c51102)

**Component Descriptions:**

1.  **RGB Image Input:** The system takes a sequence of standard RGB images as input, mimicking the data stream from a typical camera.
2.  **Feature Extraction & Tracking (Photometric Error):**  This module processes each new RGB frame. It performs two crucial tasks:
    *   **Feature Extraction (Implicit):**  Gaussian Splatting itself implicitly captures features of the scene.
    *   **Camera Pose Estimation (Tracking):**  Using photometric error minimization, this module estimates the camera's pose (position and orientation) in the scene relative to the existing 3D map. By minimizing the difference between the rendered view from the current estimated pose and the actual input image, we refine the camera pose and ensure accurate tracking.
3.  **3D Gaussian Splatting Map:** This is the core representation of the environment. It's a collection of 3D Gaussians, each defined by position, covariance, color, and opacity. These Gaussians collectively represent the scene's geometry and appearance in a continuous and differentiable manner, allowing for high-quality rendering and efficient optimization.
4.  **Covisibility Control & Image Selection:** To maintain efficiency and ensure map quality, this module selects a subset of keyframes (images) from the input sequence.  It prioritizes images that are "covisible" with the current view (i.e., images that observe overlapping parts of the scene) and are diverse enough to provide new information, preventing redundant processing and focusing on expanding and refining the map effectively.
5.  **Gaussian Splatting Optimization & Map Update:**  Based on the tracked camera poses and the selected keyframes, this module optimizes the parameters of the 3D Gaussian Splatting map. This optimization refines the position, shape, color, and opacity of the Gaussians to better represent the observed scene, leading to continuous improvement of the 3D reconstruction.  This is where the map is incrementally built and refined over time as new frames are processed.
6.  **Rendering:**  Finally, the optimized Gaussian Splatting map can be rendered from any viewpoint to generate photo-realistic images of the reconstructed scene. This allows for visualizing the map and evaluating the quality of the reconstruction.

**Data Flow:** The system operates in a loop: new RGB images are fed in, the camera pose is estimated, the map is updated based on new observations, and the map can be rendered for visualization. The covisibility control ensures efficient processing and map refinement by strategically selecting images for optimization.

## Results: Visualizing Photo-realistic 3D Reconstructions

The following videos showcase the 3D scene representations generated by our Gaussian Splatting SLAM system. These results were obtained after approximately 20 minutes of training on 1-minute RGB video sequences captured in various environments (room, kitchen, and living room).  These demonstrations highlight the model's ability to generalize and create compelling reconstructions even for larger and more complex scenes beyond the initial training data.

**Demonstration Videos:**

<table>
    <tr>
        <td>**Room Scene Reconstruction**<br><video src="https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM/assets/61759069/3873ef02-11ca-4fdb-bbb8-a02bf7c55339" width="320" height="240" controls></video></td>
        <td>**Kitchen Scene Reconstruction**<br><video src="https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM/assets/61759069/efa44483-a665-41ca-8e2f-37018e24aff4" width="320" height="240" controls></video></td>
    </tr>
</table>

**Living Room Scene - Larger Scene Generalization**<br>
<video src="https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM/assets/61759069/91978c1e-f757-4e60-9f89-8a4325a594fb" width="100%" controls></video>

**Key Observations from the Results:**

*   **Photo-realism:** The videos clearly show the high level of visual fidelity achieved by Gaussian Splatting. The reconstructions capture detailed geometry and realistic textures, resulting in visually appealing 3D models.
*   **Dense Reconstructions:** Unlike sparse point cloud SLAM, our system generates dense and complete 3D models of the environment, filling in details and creating a continuous surface representation.
*   **Generalization Capability:** The living room example demonstrates the model's ability to generalize to larger and more complex scenes than those it was initially trained on. This suggests the potential for scaling the system to map even larger environments.

While quantitative metrics are important for rigorous evaluation, these qualitative results visually demonstrate the potential of Gaussian Splatting for creating high-quality 3D maps in a SLAM setting using only RGB video. Further research can focus on quantifying the accuracy and robustness of the system compared to other SLAM methods.

## Getting Started: Installation and Setup

Ready to try out Gaussian Splatting SLAM? Follow these steps to get started:

### Prerequisites

*   **Python:**  Python 3.7 or higher is required. We recommend using a virtual environment (e.g., `venv` or `conda`) to manage dependencies.
*   **CUDA-compatible GPU:** While CPU execution is possible, a CUDA-compatible NVIDIA GPU is **highly recommended** for efficient training and rendering, especially for Gaussian Splatting models.

### Installation Steps

1.  **Install NerfStudio Base:**

    Our project is built upon the NerfStudio library. Begin by installing the base NerfStudio framework. Follow the comprehensive installation guidelines provided in the official [NerfStudio documentation](https://docs.nerf.studio/quickstart/installation.html). This will guide you through setting up your environment and installing the core NerfStudio packages.

2.  **Clone the Gaussian Splatting SLAM Repository:**

    Clone this GitHub repository to your local machine using Git:

    ```bash
    git clone https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM.git
    cd Gaussian_Splatting_SLAM
    ```

3.  **Install the `gaussian_splatting_slam` Package:**

    Navigate into the cloned repository directory and install the `gaussian_splatting_slam` package in editable mode using `pip`:

    ```bash
    python -m pip install -e .
    ```
    Then, install the NerfStudio CLI (Command Line Interface) tools:
    ```bash
    ns-install-cli
    ```

4.  **Verify Installation:**

    To confirm that the installation was successful and that the `gaussian_splatting_slam` subcommand is correctly registered with NerfStudio, run the help command:

    ```bash
    ns-train -h
    ```

    Check the output list of subcommands. You should see `gaussian_splatting_slam` listed, indicating that your custom SLAM method is now integrated into your NerfStudio environment.

### Troubleshooting Installation

*   **CUDA Issues:** If you encounter errors related to CUDA, ensure that you have a compatible NVIDIA GPU and that the CUDA Toolkit and drivers are correctly installed. Refer to the NVIDIA documentation for your specific GPU and operating system.
*   **Dependency Conflicts:** If you have existing Python environments, dependency conflicts might arise. Creating a fresh virtual environment (using `venv` or `conda`) specifically for this project is highly recommended to isolate dependencies and avoid conflicts.
*   **NerfStudio Documentation:** For general installation issues related to NerfStudio itself, consult the detailed [NerfStudio documentation](https://docs.nerf.studio/). It provides extensive troubleshooting tips and solutions for common problems.

If you encounter persistent issues, please check the repository's issue tracker or reach out for assistance.

## Usage: Training the Model on Your Data

Once installed, you can train the Gaussian Splatting SLAM model on your own RGB video datasets.

**Basic Training Command:**

To start training, use the `ns-train` command with the `gaussian_splatting_slam` method and specify the path to your data folder:

```bash
ns-train gaussian_splatting_slam --data <path_to_your_data_folder>