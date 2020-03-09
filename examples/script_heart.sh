#!bin/bash

# run jobs
echo "Starting at" `date`
for file in {train_1, train_32, train_64, train_128} do
    julia /home/asd/cmf/examples/script_heart.jl $file 10 30 &
    julia /home/asd/cmf/examples/script_heart.jl $file 20 15 &
    wait
    echo "Finished" $file "at" `date`
done