# Minimal Instructions for Training

> Two deep neural networks interact to detect antennal grooming in _Drosophila_

The system has been used on the following workstation:
| | |
|-|-|
|Processor	| 12th Gen Intel(R) Core(TM) i7-12700K   3.60 GHz|
|RAM | 16.0 GB (15.7 GB usable)|
| OS | Windows 10 Education|
| GPU | Nvidia RTX A4000, 16GB |
| Conda | 4.13|
| python | 3.7.4|
| Other | see `fly-grooming-requirements.txt`



**Network 1** is a fully convolutional network trained to detect the position of the fly's head. **Network 2** is a convolutional network with a pre-trained ResNet50 backbone trained to recognized 'antennal grooming' from frames cropped at the level of the fly's head.

## Network 1

### Preprocessing

The train set is generated using an _ImageJ_ macro. Briefly, clicking on the head of the fly (center/front) generates a circle (radius = 24 pixels) covering the head and the space in front of the head (where the legs are placed during grooming episodes). The frames are extracted from the original movies (no cropping/resizing). This process generates two sets of data: frames + labels (i.e. 1 bit masks).

> #### Output:
> - Frames
> - Masks (`.png` file, 1-bit labels)
> - Coordinates (`.csv` file)
> - log file (`.csv`, id_movie, frame number, ROI coordinates, ROI_size, if resize applied,	tool used to trace the ROI)
> - metadata file (`.csv`, info about cropping, ROI size. This is used by the macro to enable multi-session labelling)

### Training / Validation

The parameters needed to run the training are stored in a metadata file: `src\lib\metadata\metadata_fly_grooming_train_val.json`. This JSON file indicates the path where to find the Train and Validation folders with the images for training and validation. The files is also used to store the location of the trained weight file (`checkpoint`) and the result (e.g. accuracy) of training and validation.

The frames for train/validation/test must be placed in a directory with the following structure:

<hr>

/ train / grooming

/ train / no_grooming

/ validation / grooming

/ validation / no_grooming

/ test / grooming

/ test / no_grooming

<hr>

Input frames and masks are loaded by a custom dataset loader (`custom_dataset.py`). Both input frames and masks are resized to **320 x 256 pixels**. A series of transformations (flip-H, flip-V, Blur, Noise) is applied to the frames during training. By default all frames are rescaled to 0-1 (i.e. the frame is divided by 255). The model with the best validation accuracy (expressed as minimal average distance between target and predicted center of the score map) is saved automatically.

Once trained the network outputs a score-map (i.e. pixel-level predictions). A **hard-coded threshold (= 150)** is used to define the ROI of the score-map (i.e. every pixel whose value >= 150 is part of the foreground). The centroid (approximated by the max value) of the score map is used as estimate for the position of the head on the frame.

**output**: Trained model/weights

### Test

The script `test_network_1` is used to test the performance of the trained network 1 
As of 2021-03-19 the accuracy (mean distance between centroids: estimated vs labeled) on the test set is:

|Number of samples|Mean distance (+/- Std, pixels)|Pixel accuracy threshold|Accuracy|
|---|---|---|---|
| 196 | 2.73 +/- 1.12 | 8 | 99.49 %|

<br>

## Network 2

### Pre-processing

Network2 is trained on frames that are already pre-processed for grooming detection: a single image resulting from the projection along the z-axis of a 3-frame stack. The projection is performed with the standard deviation function.

To get this pre-processed file, the script `generate_dataset_from_labels.py` requires 3 input files: an excel file (`.xls`) containing the manually labeled grooming episodes (exported from Ethovision), the coordinates (`.csv`) of the centroid of the head (obtained from Network1), and the raw video file (`.avi`).

### Training

the JSON file contains the parameter needed to train the network, including the location of the train/validation folders `metadata_fly_grooming_train_network_2.json`

The whole dataset is split into separate folders:

<hr>

/ train / grooming

/ train / no_grooming

/ validation / grooming

/ validation / no_grooming

<hr>

Pytorch's data loader is used to automatically assign labels (by means of folders' name) to the frames.
> **Output**: trained model/weights

<hr>

# Minimal instructions for running inference


## Step-by-step procedure

These steps run each script individually, in a sequence. Instructions for a streamlined pipeline can be found in the next section.

- Move video files to be analyzed to the folder: `source_videos`
- Start a Miniconda3 prompt (either _powershell_ or _command line_)
- Run as Administrator

1. Type the following to activate the correct python environment and move to the networks' directory: 

```bash
conda activte pyt
cd: d:\Automatic-Grooming-Detection\src\inference\`
```

2. Run the network to find the head coordinates:

```bash
python inference_network_1.py
```

3. Run the network to detect grooming:


```bash
python inference_network_2.py
```

4. To merge the output of the two networks [speed, coordinates, p(grooming), Light ON/OFF] type:

```bash
python consolidate_raw_data.py
```
The output of `consolidate_raw_data.py` are separate _.csv_ files (i.e. individual subjects) generated inside the folder `raw_data_output`

5. videos for presentations can be generated with the script `export_movie_for_presentations.py`. This generates a PIP (picture in Picture) video with the original movie, the inset of the head, label for grooming onset/offset. This script requires:
    - The original video and the corresponding file from the `raw_data_output folder`

## Pipeline (2023-02-13)

This pipeline removes much of the typing:

- Place raw videos in the `source_videos` folder
- in a command prompt window (using the `pyt` conda environment) type: 

```
python run_networks.py BEFORE
```
or 
```
python run_networks.py AFTER
```
depending on the experimental group (Before or After)

- To analyze the data after the videos have been processed by the two networks, type in the command prompt:

```
cd ../analysis_scripts
python data_analysis.py
```