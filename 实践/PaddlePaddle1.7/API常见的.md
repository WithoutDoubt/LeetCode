#### 预测模型的存储和使用

```python
paddle.fluid.io.save_inference_model(dirname,          # 存储地址
                                     feeded_var_names, # 输入
                                     target_vars,      # 输出
                                     executor, 
                                     main_program=None, 
                                     model_filename=None, 
                                     params_filename=None, 
                                     export_for_deployment=True, 
                                     program_only=False)

```

-

```python
fluid.io.load_inference_model( dirname, 
                              executor, 
                              model_filename=None, 
                              params_filename=None, 
                              pserver_endpoints=None)
返回的是：[program，feed_target_names, fetch_targets]  # list


# dirname : 待加载模型的存储路径
# executor : 执行引擎
# model_filename : 存储
# params_filename : 存储模型参数的文件名称
# pserver_endpoints : 分布式的时候才用

# program。此处它被用于预测，因此可被称为Inference Program
# feed_target_names (list) 字符串列表，即所有输入变量的名称
# fetch_targets (list) 包含着模型的所有输出变量。通过这些输出变量即可得到模型的预测结果


#############################################
###   example  ##############################
[inference_program,                                            
 feed_target_names,                                            
 fetch_targets] = fluid.io.load_inference_model(model_save_dir,
                                                    infer_exe)     

```

