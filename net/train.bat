@REM Training of the NN
python main.py --net=ffa --crop --crop_size=240 --blocks=9 --gps=3 --bs=1 --lr=0.0001 --steps=1000 --eval_step=10 

@REM Training of the InvoFFA
@REM python main.py --net=InvoFFA --crop --crop_size=240 --blocks=9 --gps=3 --bs=1 --lr=0.0001 --steps=100 --eval_step=10 


@REM For Knowledge distillation uncomment below
@REM python main_kd.py --net=ffa --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0001 --steps=500000 --eval_step=5000
