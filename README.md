<p align="center">

  <h2 align="center">SEMPI: A Database for Understanding Social Engagement in Video-Mediated Multiparty Interaction</h2>
  <p align="center">
    <a href="https://havent-invented.github.io/"><strong>Maksim Siniukov</strong></a><sup>*</sup>
    ·  
    <a href="https://yufengyin.github.io/"><strong>Yufeng Yin</strong></a><sup>*</sup>
    ·
    <strong>Eli Fast</strong>
    ·
    <strong>Yingshan Qi</strong>
    ·
    <a href="https://aaravmo.github.io"><strong>Aarav Monga</strong></a>
    ·
    <strong>Audrey Kim</strong></a>
    ·
    <a href="https://www.ihp-lab.org/"><strong>Mohammad Soleymani</strong></a>
    <br>
    University of Southern California
    <br>
    <sup>*</sup>Equal Contribution
    <br>
</p>

We present a database for automatic understanding of Social Engagement in MultiParty Interaction (SEMPI). Social engagement is an important social signal characterizing the level of participation of an interlocutor in a conversation. Social engagement involves maintaining attention and establishing connection and rapport. Machine understanding of social engagement can enable an autonomous agent to better understand the state of human participation and involvement to select optimal actions in human-machine social interaction. Recently, video-mediated interaction platforms, \eg, Zoom, have become very popular. The ease of use and increased accessibility of video calls have made them a preferred medium for multiparty conversations, including support groups and group therapy sessions. To create this dataset, we first collected a set of publicly available video calls posted on YouTube. We then segmented the videos by speech turn and cropped the videos to generate single-participant videos. We developed a questionnaire for assessing the level of social engagement by listeners in a conversation probing the relevant nonverbal behaviors for social engagement, including back-channeling, gaze, and expressions. We used Prolific, a crowd-sourcing platform, to annotate 3,505 videos of 76 listeners by three people, reaching a moderate to high inter-rater agreement of 0.693. This resulted in a database with aggregated engagement scores from the annotators. We developed a baseline multimodal pipeline using the state-of-the-art pre-trained models to track the level of engagement achieving the CCC score of 0.454. The results demonstrate the utility of the database for future applications in video-mediated human-machine interaction and human-human social skill assessment.

## Download the data
Download the labels and the extracted features from the [here](https://1drv.ms/f/s!AslyQYfPiCM4h-BVGbctULJkksm4sg?e=mDQji1). For video and audio files, fill in the [form](https://forms.gle/ULCABeNSw9LfQZF99). Extract the files to 'data/'. Use python script 'data/engagement/code/crop_face.py' to get cropped face images and extract frames from videos.
Training the models requires both the HuBERT and InceptionI3D model weights, which are included in the download file. For licensing details, please refer to the [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) and [Inception3D](https://github.com/piergiaj/pytorch-i3d/tree/master) model licenses.

## Download pretrained models
Download engagement model weights from [here](https://1drv.ms/f/s!AslyQYfPiCM4h-BVGbctULJkksm4sg?e=mDQji1), put the weights from 'model_weights/pretrained_models' to 'code/checkpoints/engagement/'
Put the weights with corresponding names from  'model_weights/' to 'code/checkpoints/hubert_base_ls960.pt'
'code/checkpoints/hubert_large_ll60k.pt'
'code/checkpoints/rgb_imagenet.ptmodels'
'data/engagement/code/shape_predictor_68_face_landmarks.dat'

After that you should get a dataset folder structure like below:
```
code
├── requirements.txt
├── data.py
├── metr.py
├── solver_base.py               
├── videotransforms.py       
├── solver_data_process.py      
├── main.py                      # Script for model training
├── eval.py                      # Script for model evaluation
├── checkpoints/
    ├── hubert_base_ls960.pt
    ├── hubert_large_ll60k.pt
    ├── rgb_imagenet.pt
    ├── shape_predictor_68_face_landmarks.dat
    └──  engagement/             # Pretrained models weights
        ├──  model_ccc_fold_0.pt 
        ...
        └──  model_ccc_fold_4.pt
└── models/
    ├── hubert.py                # Implementation of the HuBERT model
    ├── multimodal.py            # Multimodal model
    └── pytorch_i3d.py           # I3D model for video analysis
data
└── engagement/
    ├── video                    # Video clips
    ├── code/                    # Scripts for feature extraction
    │   ├── crop_face.py         # Crops face regions from videos
    │   ├── get_audio.py         # Extracts audio from videos
    │   ├── get_text.py          # Converts speech to text
    │   ├── get_video.py         # Extracts and aligns frames
    │   ├── utils.py            
    │   └── shape_predictor_68_face_landmarks.dat  # Facial landmark detection model
    ├── frame                    # Extracted video frames
    ├── aligned_frame            # Aligned frames
    ├── audio                    # Extracted audio
    ├── featopenface             # Extracted facial Action Units (AUs) using OpenFace
    ├── text                     # Extracted text features from speech
    ├── annotations_raw_clf_reg.csv  # Raw engagement scores from annotators
    ├── label_0402_fold_0        # Regression labels for fold 0
    ...
    ├── label_0402_fold_4        # Regression labels for fold 4
    └── additional_labels        # Additional labels for classification
```

## Set up enviroment for training 
Create conda enviroment with PyTorch of your CUDA version, install fairseq and other dependencies.
```
cd code
conda create --name eng_env python=3.9
conda activate eng_env
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html  # Install PyTorch for your CUDA version 
pip install -r requirements.txt
git clone https://github.com/pytorch/fairseq # Install fairseq
cd fairseq
pip install --editable ./
cd ..
```

## Model training
Use the following script to train and evaluate the model
```
python main.py --ckpt_name label_0402_fold_0 --device 1 --model_freeze part --targettype ccc --num_labels 1 --label_root label_0402_3REG_1 --activation_fn tanh --name label_0402_fold_0 --extra_dropout 1 --kfolds 1 --hidden_size 32 --weight_decay 0.01 --expnum 8 --openfacefeat 1 --openfacefeat_extramlp 1 --openfacefeat_extramlp_dim 64
```

## Model evaluation 
Use the following script to evaluate the model
```
python eval.py --ckpt_name label_0402_fold_0 --device 1 --model_freeze part --targettype ccc --num_labels 1 --label_root label_0402_3REG_1 --activation_fn tanh --name label_0402_fold_0 --extra_dropout 1 --kfolds 1 --hidden_size 32 --weight_decay 0.01 --expnum 8 --openfacefeat 1 --openfacefeat_extramlp 1 --openfacefeat_extramlp_dim 64
```

## License

OpenSense is available under an [USC Research License](LICENSE).

3rd-party components may have their respective licenses. Please contact their respective authors to obtain licenses.

## Citation
```
@inproceedings{2024SEMPI,
  author    = {Maksim Siniukov and Yufeng Yin and Eli Fast and Yingshan Qi and Aarav Monga and Audrey Kim and Mohammad Soleymani},
  title     = {{SEMPI}: A Database for Understanding Social Engagement in Video-Mediated Multiparty Interaction},
  booktitle = {Proceedings of the 2024 International Conference on Multimodal Interaction (ICMI 2024)},
  month     = {July},
  year      = {2024},
}
```
