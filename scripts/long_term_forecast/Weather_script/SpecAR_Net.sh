export CUDA_VISIBLE_DEVICES=0,1

model_name=SpecAR_Net

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --conv_layers 6 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.00001 \
  --batch_size 64 \
  --train_epochs 50

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --d_layers 1 \
  --conv_layers 6 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.00001 \
  --batch_size 64 \
  --train_epochs 50


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --d_layers 1 \
  --conv_layers 6 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.00001 \
  --batch_size 64 \
  --train_epochs 50

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --conv_layers 6 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.00001 \
  --batch_size 64 \
  --train_epochs 50

