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