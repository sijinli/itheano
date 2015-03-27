exp_name=FCJ0_act_14
DP=mem
macid=15
JT=0027_1
EP=200
BSIZE=128
run_mac=c8k${macid}

TrainRange=0-132743
TestRange=132744-162007

test_freq=1
save_name=2015_02_28_FCJ0_act_14_cosine_test_1

SP_NOTE=
EXE=/home/grads/sijinli2/pkg/anaconda/bin/python


$EXE /home/grads/sijinli2/Projects/Itheano/Python/task/train_basicbp.py --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_${exp_name} --num-epoch=${EP} --data-provider=${DP} --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_${JT}.cfg --testing-freq=${test_freq} --batch-size=${BSIZE} --train-range=${TrainRange} --test-range=${TestRange} --save-path=/opt/visal/tmp/for_sijin/Data/saved/Test/${save_name} --solver-type=basicbp 

## for HumanEva_exp_1
##Train [0, 18503)(18503), validate [18503, 35999)(17496), Test [35999, 100945)(64946)
## TrainRange=0-18502
## TestRange=18503-35998
## for HumanEva valid 
##Train [0, 14171)(14171), validate [14171, 28715)(14544), Test [28715, 93661)(64946)
## TrainRange=0-14170
## TestRange=14171-28714

## HumanEva_exp_t  FOR FULL EXPERIMENT, PAY ATTENTION######
## TrainRange=0-28714
## TestRange=14171-28714

## for exp 10
## TrainRange=0-72435
## TestRange=72436-103079


## for exp 9
## TrainRange=0-79411
## TestRange=79412-107715


## for exp 8 
## TrainRange=0-158787
## TestRange=158788-223031


## for exp 7
## TrainRange=0-318215
## TestRange=318216-416107

## # for exp 6
## TrainRange=0-1559751
## TestRange=1559752-2103095

## for exp 5 
## TrainRange=0-109423
## TestRange=109424-148731

## For exp 4
## TrainRange=0-76047
## TestRange=76048-105367

##################
## For exp 1-3
## TrainRange=0-132743
## TestRange=132744-162007
