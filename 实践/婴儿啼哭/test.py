
import os
import wave
import numpy as np
import pickle as pkl

from tqdm import tqdm
import pandas as pd

import paddle as paddle
import paddle.fluid as fluid

from PIL import Image

import math

infer_exe = fluid.Executor(place)
#声明一个新的作用域
inference_scope = fluid.core.Scope()


LABELS = ['awake', 'diaper', 'hug', 'hungry', 'sleepy', 'uncomfortable']
N_CLASS = len(LABELS)

with open('./data_test.pkl', 'rb') as f:
    raw_data = pkl.load(f)



feeder = fluid.DataFeeder(place=place, feed_list=[image])

result = {'id': [], 'label': []}

# model_save_dir = "/home/aistudio/data/hand.inference.model"
#运行时中的所有变量都将分配给新的scope
with fluid.scope_guard(inference_scope):
    #获取训练好的模型
    #从指定目录中加载模型
    [inference_program,                                            #推理Program
     feed_target_names,                                            #是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。 
     fetch_targets] = fluid.io.load_inference_model(model_save_dir,#fetch_targets：是一个列表，从中我们可以得到推断结果。model_save_dir：模型保存的路径
                                                    infer_exe)     #infer_exe: 运行 inference model的 executor    

    infer_reader = paddle.batch(
        ,batch_size = 1
    )

    infer_data = next(infer_data)
    infer_feat = numpy.array(

    )

    infer_label = numpy.array(

    )

    assert feed_target_names[0] == 'x'
    result = infer_exe.run(inference_program,
                            feed = {feed_target_names[0]:numpy.array(infer_feat)}，
                            fetch_list = fetch_targets
                            )


    for key, value in tqdm(raw_data.items()):
    # for key, value in tqdm(raw_data):    
        
        x = np.expand_dims(np.array(value), axis=1)
        
        y = infer_exe.run(program=inference_program,         #运行推测程序
                   feed={feed_target_names[0]: x},           #喂入要预测的img
                   fetch_list=fetch_targets)[0]                   #得到推测结果,  
        if len(y) == 0:
            print(key)
        else:
            y = np.mean(y, axis=0)
            y = np.argmax(y)
            pred = LABELS[y]

        result['id'].append(key)
        result['label'].append(pred)

result = pd.DataFrame(result)
result.to_csv('./submission.csv', index=False)

####################################################### API ####################
####################################################### 线性预测 ####################
inference_scope = fluid.core.Scope()
with fluid.scope_guard(inference_scope):
    # 使用 fluid.io.load_inference_model 获取 inference program desc,
    # feed_target_names 用于指定需要传入网络的变量名
    # fetch_targets 指定希望从网络中fetch出的变量名
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(
     save_dirname, exe, None, None)

    # 将feed构建成字典 {feed_target_name: feed_target_data}
    # 结果将包含一个与fetch_targets对应的数据列表
    results = exe.run(inference_program,
                            feed={feed_target_names[0]: tensor_img},
                            fetch_list=fetch_targets)
    lab = numpy.argsort(results)

    # 打印 infer_3.png 这张图片的预测结果
    img=Image.open('image/infer_3.png')
    plt.imshow(img)
    print("Inference result of image/infer_3.png is: %d" % lab[0][0][-1])