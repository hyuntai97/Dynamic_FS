# Codebase for "Dynamic Feature-Selection"

Code authors: Hyuntae Kim, Hyeryn Park 

## Reproduce results --option {default}

```
python -u run.py \
  --is_training {1} \
  --model_id {LSTM_OUR} \
  --model {LSTM_FS_final} \
  --data {ETRI} \
  --seq_len {7} \
  --label_len {3} \
  --pred_len {3} \
  --e_layers {2} \
  --d_layers {1} \
  --enc_in {256} \
  --dec_in {256} \
  --c_out {256} \
  --itr {1} \
  --gpu {6} \
  --actor_h_dim {50} \
  --learning_rate {0.01} \
  --learning_rate2 {0.000001} \
  --lamb {20} \
```

```
bash ./test22.sh
```

## Base arguments 
`is_training`: int, Flag to decide on model training [True: Training-Test / False: Test only]

`model_id`: str, Model identifier ID 

`model`: str, Model type: LSTM_FS_final(dynamic feature selection), LSTM(no feature selection)

`data`: str, Dataset type: ETRI only 

`root_path`: str, Root path of the data file

`checkpoints`: str, Location of model checkpoints

`seq_len`: int, Input sequence length

`label_len`: int, Start token length 

`pred_len`: int, Prediction sequence length

`enc_in`: int, Encoder input size

`dec_in`: int, Decoder input size 

`c_out`: int, Model output size 

`num_layers`: int, Number of layers (LSTM Cell)

`hid_dim`: int, Hidden dimension of LSTM Cell 

`train_epochs`: int, Train epochs

`batch_size`: int, Batch size of input data

`patience`: int, Early stopping patience

`learning_rate`: float, Baseline learning rate

`learning_rate2`: float, Selector network learning rate

`loss`: str, Loss function type

`gpu`: int, GPU device number
 
`inverse`: int, Flag to decide on inverse transform [1: inverse / 0: non inverse]

`scale_type`: str, Data scaler type 

`tanh`: int, 1: Apply Tanh activation function on model output logit [minmax scaling -1~1 | 0: 0~1]

`mv_avg`: bool, Flag to decide on moving average preprocessing [True: Yes / False: No]

`lamb`: float, Feature selection lambda (regularizer)

`actor_h_dim`: int, Selector hidden dimension

`loss_op`: str, Loss function type [forecasting | total]



## File Directory 
```bash 
├─── checkpoints 
│    └── MODEL_NAME 
│        └── checkpoint.pth    
│
├─── data_provider
│    ├── data_factory.py 
│    └── data_loader.py    
│
├─── exp
│    ├── exp_basic.py 
│    └── exp_main_fs2.py
│  
├─── layers
│    └── Selector.py
│
├─── models
│    ├── LSTM_FS_final.py 
│    └── LSTM.py
│
├─── notebook 
│    └── example-file.ipynb 
│
├─── result 
│    └── MODEL_NAME 
│         └── metric.txt   
│
├─── test_results 
│    └── MODEL_NAME 
│         └── N.pdf   
│
├─── utils
│    ├── metrics.py 
│    └── tools.py
│
└─── run.py
```