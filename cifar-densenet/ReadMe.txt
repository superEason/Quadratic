1、打开constants.py，输入数据集所在目录
2、新建checkpoint保存目录和summary保存目录
3、运行split.py
4、执行python main.py --gpu-id 0 1 --batch-size 128 --learning-rate 0.1 -e

查看训练实时信息：tensorboard --logdir=./summary --port 12345

得到预测结果：执行python main.py --gpu-id 0 1 --batch-size 128 -t

