exp_name=exp
DP=croppedimgcls
macid=15
JT=0033
EP=200
BSIZE=128
run_mac=c8k${macid}

# TrainRange=0-12713
# TestRange=12714-25428
# TrainRange=0-1023
# TestRange=1023-2047
# For Face++
TrainRange=0-17719
TestRange=17720-35440
test_freq=3
save_name=Noah_face_second_test



SP_NOTE=
EXE=/home/grads/sijinli2/pkg/anaconda/bin/python

$EXE /home/grads/sijinli2/Projects/Itheano/Python/task/train_basicbp.py --data-path=/opt/visal/tmp/for_sijin/Data/Face/extractedPhotoFace++/Processed/${exp_name} --num-epoch=${EP} --data-provider=${DP} --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_${JT}.cfg --testing-freq=${test_freq} --batch-size=${BSIZE} --train-range=${TrainRange} --test-range=${TestRange} --save-path=/opt/visal/tmp/for_sijin/Data/saved/Test/${save_name} --solver-type=basicbp --load-file=/opt/visal/tmp/for_sijin/Data/saved/Test/${save_name}
