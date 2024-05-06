export CUDA_VISIBLE_DEVICES=0,1

model_name=SpecAR_Net


python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWAT \
  --model $model_name \
  --data SWAT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 32 \
  --d_ff 32 \
  --e_layers 2 \
  --conv_layers 3 \
  --enc_in 51 \
  --c_out 51 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 3
