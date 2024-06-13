# Instructions for metadata

Original metadata for `Network 1` INFERENCE: it should follow this structure
```
{
    "ROOT_DIR" : "..\\..\\source_videos",
    "FILE_EXTENSION" : "avi",
    "CHECKPOINT_NAME" : "2022-06-17_fly_grooming_network_one_train",
    "THRESHOLD" : 150,
    "TRANSFORM" : 0,
    "RESIZE_FRAME" : [320, 256],
    "CROP_X":-1,
    "CROP_Y":-1,
    "MODEL" : "conv_deconv_128",
    "TARGET_DIR" : "..\\..\\results\\head_coordinates\\",
    "COMMENT":"Crop is from the top left corner!"
}
```