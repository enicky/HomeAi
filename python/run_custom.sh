export CUDA_VISIBLE_DEVICES=0

model_name=LSTM

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=16
enc_in=4
c_out=1

python -u run.py \
  --scaler StandardScaler \
  --target Watt \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./data/ \
  --data_path merged_and_sorted_file.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in $enc_in \
  --c_out $c_out \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
