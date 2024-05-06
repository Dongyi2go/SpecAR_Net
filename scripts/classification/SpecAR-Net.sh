export CUDA_VISIBLE_DEVICES=0

model_name=SpecAR_Net

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model $model_name \
  --data UEA \
  --e_layers 5 \
  --conv_layers 6 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --patience 10 \
  --logdir 'tensorboard/classification_log'

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --conv_layers 3 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 64 \
  --num_kernels 4 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10 \
  --logdir 'tensorboard/classification_log'

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Handwriting/ \
  --model_id Handwriting \
  --model $model_name --data UEA \
  --e_layers 2 \
  --conv_layers 3 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10 \
  --logdir 'tensorboard/classification_log'

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model $model_name \
  --data UEA \
  --e_layers 1 \
  --conv_layers 6 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 64 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --patience 10  \
  --logdir 'tensorboard/classification_log'

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/JapaneseVowels/ \
  --model_id JapaneseVowels \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --conv_layers 3 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 64 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0004 \
  --train_epochs 30 \
  --patience 10  \
  --logdir 'tensorboard/classification_log'

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PEMS-SF/ \
  --model_id PEMS-SF \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --conv_layers 3 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10 \
  --logdir 'tensorboard/classification_log'

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1 \
  --model $model_name \
  --data UEA \
  --e_layers 4 \
  --conv_layers 3 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10 \
  --logdir 'tensorboard/classification_log'

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP2/ \
  --model_id SelfRegulationSCP2 \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --conv_layers 6 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 64 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10 \
  --logdir 'tensorboard/classification_log'

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SpokenArabicDigits/ \
  --model_id SpokenArabicDigits \
  --model $model_name \
  --data UEA \
  --e_layers 5 \
  --conv_layers 6 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --patience 10 \
  --logdir 'tensorboard/classification_log'

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model $model_name \
  --data UEA \
  --e_layers 5 \
  --conv_layers 6 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --patience 10 \
  --logdir 'tensorboard/classification_log'
