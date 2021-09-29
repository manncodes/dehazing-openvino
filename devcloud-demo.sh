# run this bash script only on devcloud, and in a compute node.
qsub -I
conda activate pytorch
cd dehazing-openvino/net
python test.py --test_imgs test