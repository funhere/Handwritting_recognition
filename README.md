# OCR_CRNN
Use tensorflow to implement a Deep Neural Network for handwritting telephone number recognition mainly based on the paper "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition".
This model consists of a CNN stage, RNN stage and CTC loss for scene text recognition task.

## Installation
```
pip3 install -r requirements.txt

If using anaconda you can run conda env create -f environment.yml.
```

## Where the pre-trained checkpoint is saved to.
CHECKPOINT_PATH=./checkpoints/crnn/your_check_points.ckpt

## Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=./checkpoints/crnn/

## Where the dataset is saved to.
DATASET_DIR=./data/ocr/

## Download the pre-trained checkpoint.
```
if [ ! -d "$CHECKPOINT_PATH" ]; then
mkdir ${CHECKPOINT_PATH}
fi
```

## Download the dataset
```shell
python download_and_convert_data.py \
  --dataset_name=mnist \
  --dataset_dir=${DATASET_DIR}
```

## Convert a dataset to TFRecords format
```shell
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=data/ocr/train/ \
    --output_name=ocr_train \
    --output_dir=data/ocr/
```

## Train your own model
#### Train model from scratch
```shell
python train_ocr_recognition.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --dataset_name=ocr \
  --dataset_split_name=train \
  --model_name=crnn \
  --save_summaries_secs=60 \
  --save_interval_secs=600 \
  --weight_decay=0.0005 \
  --optimizer=adam \
  --learning_rate=0.01 \
  --learning_rate_decay_factor=0.95 \
  --batch_size=32
```

#### Train model from the snapshot
```shell
CHECKPOINT_PATH=./checkpoints/crnn/your_check_points.ckpt
python train_ocr_recognition.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --dataset_name=ocr \
  --dataset_split_name=train \
  --model_name=crnn \
  --save_summaries_secs=60 \
  --save_interval_secs=600 \
  --weight_decay=0.0005 \
  --optimizer=adam \
  --learning_rate=0.01 \
  --learning_rate_decay_factor=0.95 \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --batch_size=32
```

####  Fine-tune only the assigned layers
```shell
python train_ocr_recognition.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --dataset_name=ocr \
  --dataset_split_name=train \
  --model_name=crnn \
  --checkpoint_path=${CHECKPOINT_DIR}/your_check_points.ckpt \
  --checkpoint_exclude_scopes=crnn/logits \
  --trainable_scopes=crnn/logits \
  --save_summaries_secs=60 \
  --save_interval_secs=600 \
  --weight_decay=0.0005 \
  --optimizer=adam \
  --learning_rate=0.01 \
  --learning_rate_decay_factor=0.95 \
  --batch_size=32 
```

## Run evaluation
```shell
CHECKPOINT_PATH=./checkpoints/crnn/your_check_points.ckpt
python eval_ocr_recognition.py \
  --eval_dir=${EVAL_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --dataset_name=ocr \
  --dataset_split_name=test \
  --model_name=crnn \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --wait_for_checkpoints=True \
  --batch_size=1 \
  --max_num_batches=500
```

# TODO
1. train model on dataset Mnist;
2. train model on self-generate multi-digital dataset;
3. train model on GAN generate dataset.
4. try other models: attention, SSD+RNN+CTC, inception_resnet_v2, inception_v3/v4,...
5. model ensemable
