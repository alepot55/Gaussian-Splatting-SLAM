# Gaussian Splatting for SLAM

The proposed model is a dense Simultaneous Localization and Mapping (SLAM) system that aims to leverage 3D Gaussian Splatting to achieve photo-realistic reconstructions. For the last-year excellence program at Sapienza University of Rome, the project focuses on trying to develop a model for SLAM using Gaussian representations. SLAM is a critical component in robotics and computer vision, enabling robots and autonomous systems to navigate and map their environment. 

The developed method builds upon the foundation provided by NerfStudio, a powerful library for 3D reconstruction. Specifically, my model modifies NerfStudio’s Splatfacto method to adapt to SLAM systems. Unlike many SLAM models that rely on depth information, the model can operates solely on RGB images thanks to a tracking module enhancing system robustness against camera motion and occlusions from photometric error. To maintain a diverse set of images and improve efficiency, a covisibility control mechanism aims to select a subset of diverse images to balance the need for coverage with minimizing redundancy. All implemented components are highlighted image below, diasplaying an overview of the SLAM system used. On the right, the video shows the output scene representation of my custom dataset (my room) after 20 minutes of training on a 1-minute RGB video.
<p align="center">
    <table>
        <tr>
            <td>
                <img src="https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM/assets/61759069/1592bd44-b794-460f-b754-501145c51102" width="100%">
            </td>
            <td>
                <video src="https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM/assets/61759069/f829f6ac-1f62-4308-ba8b-ba1894d31344" type="video/webm"> width="320" height="240" controls>
                </video>
            </td>
            <td>
                <video src="https://github.com/alessandro-potenza/Gaussian_Splatting_SLAM/assets/61759069/2590cd32-761f-41a1-8106-289514dddde4" type="video/webm"> width="320" height="240" controls>
                </video>
            </td>
        </tr>
    </table>
</p>



Professor: _Thomas Alessandro Ciarfuglia_

## Getting Started.


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
     The `gaussian_splatting_slam` method should appear in the list of subcommands.

4. **Training the Model**:
   - Launch training using the command:
     ```
     ns-train gaussian_splatting_slam --data <data_folder>
     ```
     Replace `<data_folder>` with the path to your data folder.
   - You can use either [existing datasets](https://docs.nerf.studio/quickstart/existing_dataset.html) or [create your custom dataset](https://docs.nerf.studio/quickstart/custom_dataset.html).
