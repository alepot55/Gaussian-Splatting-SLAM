# Gaussian Splatting for SLAM

## Overview

This project presents an innovative dense Simultaneous Localization and Mapping (SLAM) system that leverages 3D Gaussian Splatting for photo-realistic reconstructions. Developed as part of the last-year excellence program at Sapienza University of Rome, this model explores the use of Gaussian representations in SLAM, a critical component in robotics and computer vision for autonomous navigation and environment mapping.

## Key Features

- Built upon NerfStudio's powerful 3D reconstruction library
- Modifies NerfStudio's Splatfacto method for SLAM applications
- Operates solely on RGB images, enhancing robustness against camera motion and occlusions
- Implements a tracking module based on photometric error for improved system robustness
- Utilizes a covisibility control mechanism for efficient and diverse image selection
- Achieves photo-realistic reconstructions in various environments

## System Architecture

Our SLAM system incorporates several key components, as illustrated in the image below:

![SLAM System Overview](https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM/assets/61759069/1592bd44-b794-460f-b754-501145c51102)

## Results

The following videos demonstrate the output scene representations of custom datasets (room, kitchen, and living room) after 20 minutes of training on 1-minute RGB videos. These results showcase the model's ability to generalize to larger scenes.

<table>
    <tr>
        <td><video src="https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM/assets/61759069/3873ef02-11ca-4fdb-bbb8-a02bf7c55339" width="320" height="240" controls></video></td>
        <td><video src="https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM/assets/61759069/efa44483-a665-41ca-8e2f-37018e24aff4" width="320" height="240" controls></video></td>
    </tr>
</table>

<video src="https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM/assets/61759069/91978c1e-f757-4e60-9f89-8a4325a594fb" width="100%" controls></video>

## Getting Started

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended)

### Installation

1. **Install NerfStudio**
   
   Follow the installation guidelines provided in the [NerfStudio documentation](https://docs.nerf.studio/quickstart/installation.html).

2. **Clone the Repository**
   
   ```
   git clone https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM.git
   cd Gaussian_Splatting_SLAM
   ```

3. **Install the Package**
   
   ```
   python -m pip install -e .
   ns-install-cli
   ```

4. **Verify Installation**
   
   ```
   ns-train -h
   ```
   
   Ensure that `gaussian_splatting_slam` appears in the list of subcommands.

## Usage

To train the model on your dataset:

```
ns-train gaussian_splatting_slam --data <path_to_your_data_folder>
```

You can use either [existing datasets](https://docs.nerf.studio/quickstart/existing_dataset.html) or [create your custom dataset](https://docs.nerf.studio/quickstart/custom_dataset.html).

## Acknowledgements

- Professor Thomas Alessandro Ciarfuglia for project supervision

