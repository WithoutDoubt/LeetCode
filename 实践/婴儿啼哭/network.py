class MyNet():
    def __init__(self, is_train=True):

        self.is_train = is_train
        self.weight_decay = 1e-4

    def net(self, input, class_dim):
    
        depth = [3, 3, 3, 3, 3]
        num_filters = [16, 16, 32, 32, 64]

        conv = self.conv_bn_layer(
            input=input, num_filters=16, filter_size=3, act='elu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1)
                conv = fluid.layers.batch_norm(input=conv)
        print(conv.shape)
        pool = fluid.layers.pool2d(
            input=conv, pool_size=2, pool_type='max', global_pooling=False)
       
        pool = fluid.layers.conv2d(
            input=pool, num_filters=32, filter_size=[3, 1], stride=[2, 1], act='elu')
        print(pool.shape)
        pool = fluid.layers.flatten(pool)
        pool = fluid.layers.dropout(pool, dropout_prob=0.5)
        net = fluid.layers.fc(input=pool,
                              size=128,
                              act="elu"
                              )
        print(net.shape)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(input=net,
                              size=class_dim,
                              act="softmax",
                              param_attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.Uniform(-stdv,
                                                                        stdv),
                                  regularizer=fluid.regularizer.L2Decay(self.weight_decay))
                              )
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      bn_init_value=1.0):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False,
            param_attr=fluid.ParamAttr(regularizer=fluid.regularizer.L2Decay(self.weight_decay)))
        return fluid.layers.batch_norm(
                input=conv, act=act, is_test=not self.is_train,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(bn_init_value),
                    regularizer=None))

    def shortcut(self, input, ch_out, stride):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride):
        conv0 = self.conv_bn_layer(
            input=input, num_filters=num_filters, filter_size=1, act='relu')
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        conv2 = self.conv_bn_layer(
            input=conv1, num_filters=num_filters * 4, filter_size=1, act=None, bn_init_value=0.0)

        short = self.shortcut(input, num_filters * 4, stride)

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')

# 定义输入输出层
image = fluid.layers.data(name='image', shape=[1, 323, 60], dtype='float32')#单通道，28*28像素值
label = fluid.layers.data(name='label', shape=[1], dtype='int64')          #图片标签

model = MyNet()
out = model.net(input=image, class_dim=N_CLASS)

# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=out, label=label) 
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=out, label=label)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=2e-4)   #使用Adam算法进行优化
opts = optimizer.minimize(avg_cost)

# 定义一个使用CPU的解析器
model_save_dir = "/home/aistudio/data/hand.inference.model"

place = fluid.CUDAPlace(0)                             # place = fluid.CPUPlace() CPU
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
# fluid.io.load_params(executor=exe, dirname=model_save_dir,
#                     main_program=None)
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

def draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost) 
    plt.plot(iters, accs,color='green',label=lable_acc) 
    plt.legend()
    plt.grid()
    plt.show()
    
all_train_iter=0
all_train_iters=[]
all_train_costs=[]
all_train_accs=[]

from tqdm import tqdm

EPOCH_NUM=20  # 调参 训练轮数

for pass_id in range(EPOCH_NUM):
    # 进行训练
    for data in tqdm(train_reader()):                         #遍历train_reader
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),#运行主程序
                                        feed=feeder.feed(data),              #给模型喂入数据
                                        fetch_list=[avg_cost, acc])          #fetch 误差、准确率  
                                        
        all_train_iter=all_train_iter+1
        all_train_iters.append(all_train_iter)
        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])

    print('Pass:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, np.mean(train_cost), np.mean(train_acc)))

    # 进行测试
    test_accs = []
    test_costs = []
    #每训练一轮 进行一次测试
    for batch_id, data in enumerate(valid_reader()):                         #遍历test_reader
        test_cost, test_acc = exe.run(program=fluid.default_main_program(), #执行训练程序
                                      feed=feeder.feed(data),               #喂入数据
                                      fetch_list=[avg_cost, acc])           #fetch 误差、准确率
        test_accs.append(test_acc[0])                                       #每个batch的准确率
        test_costs.append(test_cost[0])                                     #每个batch的误差

    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))                         #每轮的平均误差
    test_acc = (sum(test_accs) / len(test_accs))                            #每轮的平均准确率
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))
    
    #保存模型
    # 如果保存路径不存在就创建
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print ('save models to %s' % (model_save_dir))
    fluid.io.save_inference_model(model_save_dir,  #保存推理model的路径
                                  ['image'],       #推理（inference）需要 feed 的数据
                                  [out],       #保存推理（inference）结果的 Variables
                                  exe)             #executor 保存 inference model
draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"trainning cost","trainning acc")        