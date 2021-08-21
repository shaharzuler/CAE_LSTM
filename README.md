# CAE-LSTM
A prototype implementation for [In-air Knotting of Rope using Dual-Arm Robot based on Deep Learning](https://arxiv.org/pdf/2103.09402.pdf), using PyTorch-Lightning.

The simulation environment and the sample data are taken from [OpenAI Gym Car Racing environment](https://gym.openai.com/envs/CarRacing-v0/).

Genrating training data was done by using [this RL agent](https://github.com/xtma/pytorch_car_caring).

A small sample of this dataset is provided under the "sample_dataset" directory.
### End to end continuous inference:
![](infer_all.gif)

******
### Install requirements 
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### Training
In order to train the convolutional autoencoder:
```
python train_autoencoder.py -img_size "96,96,1" -root_folder "sample_dataset"
```
In order to train the LSTM:
```
python train_lstm.py -img_size "96,96,1" -root_folder "sample_dataset"
```

### Inference
In order to perform an end-to-end continuous inference:
```
python infer_CAE_LSTM.py -img_size "96,96,1" -root_folder "sample_dataset" -trained_encoder_path "sample_dataset/autoencoder_logs/lightning_logs/version_0/checkpoints/epoch=132-step=5186.ckpt" -trained_lstm_path "sample_dataset/lstm_logs/lightning_logs/version_1/checkpoints/epoch=0-step=1084.ckpt"
```
In order to visualize inference of the convolutional autoencoder:
```
python inference_utils/infer_autoencoder.py -img_size "96,96,1" -root_folder "sample_dataset" -trained_encoder_path "sample_dataset/autoencoder_logs/lightning_logs/version_0/checkpoints/epoch=132-step=5186.ckpt"
```
In order to check inference of the LSTM:
```
python inference_utils/infer_lstm.py -img_size "96,96,1" -root_folder "sample_dataset" -trained_encoder_path "sample_dataset/autoencoder_logs/lightning_logs/version_0/checkpoints/epoch=132-step=5186.ckpt" -trained_lstm_path "sample_dataset/lstm_logs/lightning_logs/version_1/checkpoints/epoch=0-step=1084.ckpt"
```


*****
### Future work:
- Try a convolutional variational autoencoder to see if it performs better.
- Normalize LSTM loss by sequence length
- Normalize convolutional autoencoder loss by image size