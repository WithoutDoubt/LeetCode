import paddle.fluid as fluid
import paddle
import reader
import network
import numpy as np
import os
from config import train_parameters, init_train_parameters
init_train_parameters()

file_list = os.path.join(train_parameters['data_dir'], "eval.txt")
with fluid.dygraph.guard():
    model, _ = fluid.dygraph.load_dygraph("save_dir/params")
    alexnet = network.AlexNet("alexnet", train_parameters['class_dim'])
    alexnet.load_dict(model)
    alexnet.eval()
    test_reader = paddle.batch(reader.custom_image_reader(file_list, reader.train_parameters['data_dir'], 'val'),
                            batch_size=1,
                            drop_last=True)
    accs = []
    for batch_id, data in enumerate(test_reader()):
        dy_x_data = np.array([x[0] for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).astype('int')
        y_data = y_data[:, np.newaxis]
        
        img = fluid.dygraph.to_variable(dy_x_data)
        label = fluid.dygraph.to_variable(y_data)
        label.stop_gradient = True

        out, acc = alexnet(img, label)
        accs.append(acc.numpy()[0])
print(np.mean(accs))