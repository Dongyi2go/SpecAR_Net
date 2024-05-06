export CUDA_VISIBLE_DEVICES=0,1

model_name=SpecAR_Net

python -u run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model $model_name \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2\
  --conv_layers 4 \
  --enc_in 25 \
  --c_out 25 \
  --learning_rate 0.0001 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3
