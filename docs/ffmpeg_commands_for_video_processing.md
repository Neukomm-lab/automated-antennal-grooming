# Processing video files with ffmpeg

## Clip video in time
```
ffmpeg -ss 00:00:00.0 -i video_file_IN.avi -c copy -t 00:00:40.0 video_file_IN.avi
```

## Clip and downscale video
```
ffmpeg -ss 00:00:25.0 -i video_file_IN.avi -vf scale=800:600 -t 00:00:50.0 video_file_IN.avi
```

## Output: Picture in Picture (PIP) video
```
ffmpeg -i video_file_1_IN.avi -i video_file_2_IN.avi -filter_complex "[1]scale=iw*2:ih*2 [pip]; [0][pip] overlay=main_w-overlay_w-10:main_h-overlay_h-10" -vcodec h264 -acodec libvo_aacenc video_file_IN.avi
```