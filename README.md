# Video classification example with Inception and LSTM:

Video classification example with Inception and LSTM. See detailed description in this blog post: https://www.apriorit.com/dev-blog/609-ai-long-short-term-memory-video-classification

1. Place the videos from your dataset in data/train and data/test folders. Each video type should have its own folder

>	| data/test
> >		| Football
> >		| Commercial
> >		...
>	| data/train
> >		| Football
> >		| Commertial
> >		...

2. Extract files from video with script extract_files.py. Pass video files extenssion as a param

`	$ python extract_files.py mp4`

3. Check the data_file.csv and choose the acceptable sequence length of frames. It should be less or equal to lowest one if you want to process all videos in dataset.
4. Extract sequence for each video with InceptionV3 and train LSTM. Run train.py script with sequence_length, class_limit, image_height, image_width args

`	$ python train.py 75 2 720 1280`

5. Save your best model file. (For example, lstm-features.hdf5)
6. Use clasify.py script to clasify your video. Args sequence_length, class_limit, saved_model_file, video_filename

`	$ python clasify.py 75 2 lstm-features.hdf5 video_file.mp4`

The result will be placed in result.avi file.

## Requirements

This code requires you have Keras 2 and TensorFlow 1 or greater installed. Please see the `requirements.txt` file. To ensure you're up to date, run:

`pip install -r requirements.txt`

You must also have `ffmpeg` installed in order to extract the video files.
