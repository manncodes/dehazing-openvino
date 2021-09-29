# run this bash script only on devcloud, and in a compute node.

#start an interactive job shell
qsub -I -l nodes=1:gpu:ppn=2

# clone pytorch conda env for devcloud (done due to no root access)
conda create --name user_pytorch --clone pytorch

# activate cloned conda env
conda activate user_pytorch

# move into the net folder where all the main repices of the project are!
cd net

#test script
python test.py --test_imgs test

# Runs successfully on Jupyter HUB of intel