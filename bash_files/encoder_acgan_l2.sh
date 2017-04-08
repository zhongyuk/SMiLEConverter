cd ..
python main_encoder.py --gen_logs_dir=/logs/acgan_l2/ --logs_dir=encoder_logs/acgan_l2/ --num_iter=40000 --learning_rate=2e-4 --optimizer_param=0.5  --iterations=1e5 --mode=train
