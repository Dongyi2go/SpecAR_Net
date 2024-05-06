export CUDA_VISIBLE_DEVICES=0,1

model_name=SpecAR_Net

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
  --model_id MSL \
  --model $model_name \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 8 \
  --d_ff 8 \
  --e_layers 5 \
  --conv_layers 6 \
  --enc_in 55 \
  --c_out 55 \
  --win_len 100 \
  --n_fft  100 \
  --win_func  hamming \
  --learning_rate 0.005 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 1
