from config import train_parameters, logger
import paddle.fluid as fluid
from network import AlexNet
import numpy as np
import paddle
import reader
import math
import os

momentum_rate = 0.9
l2_decay = 1.2e-4


def optimizer_setting(params, parameter_list):
    ls = params["learning_strategy"]
    if "total_images" not in params:
        total_images = 6149
    else:
        total_images = params["total_images"]

    batch_size = ls["batch_size"]
    step = int(math.ceil(float(total_images) / batch_size))
    bd = [step * e for e in ls["epochs"]]
    lr = params["lr"]
    num_epochs = params["num_epochs"]
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.cosine_decay(
            learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
        momentum=momentum_rate,
        regularization=fluid.regularizer.L2Decay(l2_decay),
        parameter_list=parameter_list)

    return optimizer


epoch_nums = train_parameters["num_epochs"]
batch_size = train_parameters["train_batch_size"]
def train():
    with fluid.dygraph.guard():
        alexnet = AlexNet("alexnet", reader.train_parameters['class_dim'])
        optimizer = optimizer_setting(train_parameters, alexnet.parameters())
        file_list = os.path.join(reader.train_parameters['data_dir'], "train.txt")
        train_reader = paddle.batch(reader.custom_image_reader(file_list, reader.train_parameters['data_dir'], 'train'),
                                batch_size=batch_size,
                                drop_last=True)
        
        if train_parameters["continue_train"]:
            # 加载上一次训练的模型，继续训练
            model, _ = fluid.dygraph.load_persistables("save_dir")
            se_resnext.load_dict(model)

        for epoch_num in range(epoch_nums):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int')
                y_data = y_data[:, np.newaxis]
                
                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True

                out, acc = alexnet(img, label)
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)

                # dy_out = avg_loss.numpy()
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                alexnet.clear_gradients()

                dy_param_value = {}
                for param in alexnet.parameters():
                    dy_param_value[param.name] = param.numpy

                if batch_id % 10 == 0:
                    logger.info("Loss at epoch {} step {}: {}, acc: {}".format(epoch_num, batch_id, avg_loss.numpy(), acc.numpy()))
                    fluid.dygraph.save_dygraph(alexnet.state_dict(), "save_dir/params")
        logger.info("Final loss: {}".format(avg_loss.numpy()))


if __name__ == "__main__":
    train()