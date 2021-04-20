# Video Captioning using Deep Learning

## Frameworks and Libraries Used
* Tensorflow 2.3
* Numpy 1.8
* CV2 4.2
* NLTK 3.5 (Natural Language ToolKit)

## How to run it to generate captions
* Download our trained model from [this link](https://drive.google.com/drive/folders/1U_fCpjMGgf6lotg7mZYqlgYwPav60POK?usp=sharing), mainly encoder_model.h5, decoder_model_weights.h5, tokenizer.json and save them in directory `saved_models/`.
* For repeatable use, download VGG weights from [here](https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5).
* import `Video_Caption_Generator` class and create its object.
```
from inference import Video_Caption_Generator
vcg = Video_Caption_Generator()
```
* specify the path of the video and number of captions to generate as beam_width. For Example:
```
path = r'C:\Users\Yogesh\Desktop\Projects\Video Captioning\dataset\demo\11.gif'
beam_width = 5
```
* generate captions using `get_video_captions()`
```
captions = vcg.get_video_captions(path, beam_width)
```

## To train the model
* Download the preprocessed dataset from [here](https://www.dropbox.com/sh/whatkfg5mr4dr63/AACKCO3LwSsHK4_GOmHn4oyYa?dl=0) and save them in directory `dataset/`.
* import `Video_Captioning_Model` class and create its object.
```
from training import Video_Captioning_Model
vcm = Video_Captioning_Model()
```
* change number of epochs to train at line _120_ and train using:
```
vcm.train()
```

## To test the results
* import `Video_Caption_Generator` class and create its object.
```
from inference import Video_Caption_Generator
vcg = Video_Caption_Generator()
```
* Specify the beam widths on which you want to test. For Example:
```
beam_width_values = [1, 2, 3, 4, 5, 10, 15, 20]
```
* use `test()` method. This will return _bleu_ and _meteor_ score of generated captions. The scores are generated using nltk library.
```
bleu, meteor = vcg.test(beam_width_values)
```

## Our results

| Beam Width | Bleu | Meteor |
|------------|------|--------|
|1  |0.1354  | 0.4695 |
|2  |0.1716  | 0.5082 |
|3  |0.2010  | 0.5360 |
|4  |0.2197  | 0.5503 |
|5  |0.2367  | 0.5645 |
|10 |0.2803  | 0.6093 |
|15 |0.3283  | 0.6368 |
|20 |0.3469  | 0.6509 |