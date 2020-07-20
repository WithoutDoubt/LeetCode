# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/4/25 21:19
'''

from model.MetaRNNCells import MetaRNNCell ,MetaRNNCellBase, MetaLSTMCell

import torch
from torch.nn import Module
from torch.autograd import Variable

class MetaRNNBase(Module):
    def __init__(self, mode, input_size, hidden_size, hyper_hidden_size, hyper_embedding_size, num_layers, bias=True, bias_hyper=True, gpu=True, bidirectional=False):
        super(MetaRNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embedding_size = hyper_embedding_size
        self.num_layers = num_layers
        self.bias = bias
        self.bias_hyper = bias_hyper
        self.gpu = gpu
        self.bidirectional=bidirectional

        mode2cell = {'MetaRNN': MetaRNNCell,
                     'MetaLSTM': MetaLSTMCell}

        Cell = mode2cell[mode]

        kwargs = {'input_size': input_size,
                  'hidden_size': hidden_size,
                  'hyper_hidden_size': hyper_hidden_size,
                  'hyper_embedding_size': hyper_embedding_size,
                  'bias': bias,
                  'bias_hyper': bias_hyper}

        if self.bidirectional:
            self.cell0 = Cell(**kwargs)
            for i in range(1, num_layers):
                kwargs['input_size'] = hidden_size * 2
                cell = Cell(**kwargs)
                setattr(self, 'cell{}'.format(i), cell)

            kwargs['input_size'] = input_size
            self.cellb0 = Cell(**kwargs)
            for i in range(1, num_layers):
                kwargs['input_size'] = hidden_size * 2
                cell = Cell(**kwargs)
                setattr(self, 'cellb{}'.format(i), cell)
        else:
            self.cell0 = Cell(**kwargs)
            for i in range(1, num_layers):
                kwargs['input_size'] = hidden_size
                cell = Cell(**kwargs)
                setattr(self, 'cell{}'.format(i), cell)

    def _initial_states(self, inputSize):
        main_zeros = Variable(torch.zeros(inputSize, self.hidden_size))
        meta_zeros = Variable(torch.zeros(inputSize, self.hyper_hidden_size))
        if self.gpu:
            main_zeros = main_zeros.cuda()
            meta_zeros = meta_zeros.cuda()
        zeros = (main_zeros, meta_zeros)
        if self.mode == 'MetaLSTM':
            states = [((main_zeros, main_zeros), (meta_zeros, meta_zeros)), ] * self.num_layers
        else:
            states = [zeros] * self.num_layers
        return states
    
    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on all_weights, that's why the
                # no_grad() is necessary.
                torch._cudnn_rnn_flatten_weight(
                    all_weights, (4 if self.bias else 2),
                    self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))    

    def forward(self, input, length=None):
        states = self._initial_states(input.size(0))
        outputs = []
        time_steps = input.size(1)

        if length is None:
            length = Variable(torch.LongTensor([time_steps] * input.size(0)))
            if self.gpu:
                length = length.cuda()

        if self.bidirectional:
            states_b = self._initial_states(input.size(0))
            outputs_f = []
            outputs_b = []
            hx = None

            for num in range(self.num_layers):
                for t in range(time_steps):
                    x = input[:, t, :]
                
                    if self.mode.startswith('MetaLSTM'):
                        (main_h, main_c), (meta_h, meta_c) = getattr(self, 'cell{}'.format(num))(x, states[num])
                        mask_main_h = (t < length).float().unsqueeze(1).expand_as(main_h)
                        mask_main_c = (t < length).float().unsqueeze(1).expand_as(main_c)
                        mask_meta_h = (t < length).float().unsqueeze(1).expand_as(meta_h)
                        mask_meta_c = (t < length).float().unsqueeze(1).expand_as(meta_c)
                        main_h = main_h * mask_main_h + states[0][0][0] * (1 - mask_main_h)
                        main_c = main_c * mask_main_c + states[0][0][1] * (1 - mask_main_c)
                        meta_h = meta_h * mask_meta_h + states[0][1][0] * (1 - mask_meta_h)
                        meta_c = meta_c * mask_meta_c + states[0][1][1] * (1 - mask_meta_c)
                        states[num] = ((main_h, main_c),(meta_h, meta_c))
                        outputs_f.append(main_h)
                    else:
                        main_h, meta_h = getattr(self, 'cell{}'.format(num))(x, states[num])
                        mask_main_h = (t < length).float().unsqueeze(1).expand_as(main_h)
                        mask_meta_h = (t < length).float().unsqueeze(1).expand_as(meta_h)
                        main_h = main_h * mask_main_h + states[0][0] * (1 - mask_main_h)
                        meta_h = meta_h * mask_meta_h + states[0][1] * (1 - mask_meta_h)
                        states[num] = (main_h, meta_h)
                        outputs_f.append(main_h)
                for t in range(time_steps)[::-1]:
                    x = input[:, t, :]
                    if self.mode.startswith('MetaLSTM'):
                        (main_h, main_c), (meta_h, meta_c) = getattr(self, 'cell{}'.format(num))(x, states_b[num])
                        mask_main_h = (t < length).float().unsqueeze(1).expand_as(main_h)
                        mask_main_c = (t < length).float().unsqueeze(1).expand_as(main_c)
                        mask_meta_h = (t < length).float().unsqueeze(1).expand_as(meta_h)
                        mask_meta_c = (t < length).float().unsqueeze(1).expand_as(meta_c)
                        main_h = main_h * mask_main_h + states_b[0][0][0] * (1 - mask_main_h)
                        main_c = main_c * mask_main_c + states_b[0][0][1] * (1 - mask_main_c)
                        meta_h = meta_h * mask_meta_h + states_b[0][1][0] * (1 - mask_meta_h)
                        meta_c = meta_c * mask_meta_c + states_b[0][1][1] * (1 - mask_meta_c)
                        states_b[num] = ((main_h, main_c),(meta_h, meta_c))
                        outputs_b.append(main_h)
                    else:
                        main_h, meta_h = getattr(self, 'cell{}'.format(num))(x, states_b[num])
                        mask_main_h = (t < length).float().unsqueeze(1).expand_as(main_h)
                        mask_meta_h = (t < length).float().unsqueeze(1).expand_as(meta_h)
                        main_h = main_h * mask_main_h + states_b[0][0] * (1 - mask_main_h)
                        meta_h = meta_h * mask_meta_h + states_b[0][1] * (1 - mask_meta_h)
                        states_b[num] = (main_h, meta_h)
                        outputs_b.append(main_h)
                    
                outputs_b.reverse()
                input = torch.cat([torch.stack(outputs_f).transpose(0, 1), torch.stack(outputs_b).transpose(0, 1)], 2)
                outputs_f = []
                outputs_b = []
            # output = input, input[-1]
        else:
            
            outputs_f = []
            for num in range(self.num_layers):
                for t in range(time_steps):
                    # print("========== num:{},t:{}".format(num,t))
                    x = input[:, t, :]
                    if self.mode.startswith('MetaLSTM'):
                        (main_h, main_c), (meta_h, meta_c) = getattr(self, 'cell{}'.format(num))(x, states[num])
                        mask_main_h = (t < length).float().unsqueeze(1).expand_as(main_h)
                        mask_main_c = (t < length).float().unsqueeze(1).expand_as(main_c)
                        mask_meta_h = (t < length).float().unsqueeze(1).expand_as(meta_h)
                        mask_meta_c = (t < length).float().unsqueeze(1).expand_as(meta_c)
                        main_h = main_h * mask_main_h + states[0][0][0] * (1 - mask_main_h)
                        main_c = main_c * mask_main_c + states[0][0][1] * (1 - mask_main_c)
                        meta_h = meta_h * mask_meta_h + states[0][1][0] * (1 - mask_meta_h)
                        meta_c = meta_c * mask_meta_c + states[0][1][1] * (1 - mask_meta_c)
                        states[num] = ((main_h, main_c),(meta_h, meta_c))
                        outputs_f.append(main_h)
                    else:
                        main_h, meta_h = getattr(self, 'cell{}'.format(num))(x, states[num])
                        mask_main_h = (t < length).float().unsqueeze(1).expand_as(main_h)
                        mask_meta_h = (t < length).float().unsqueeze(1).expand_as(meta_h)
                        main_h = main_h * mask_main_h + states[0][0] * (1 - mask_main_h)
                        meta_h = meta_h * mask_meta_h + states[0][1] * (1 - mask_meta_h)
                        states[num] = (main_h, meta_h)
                        outputs_f.append(main_h)
              
                # print("++++++++")
                outputs_f = tuple(outputs_f)
                # print("outputs_f shape is {}".format(outputs_f.size))


                input = torch.stack(outputs_f).transpose(0,1)
                

                
                
                # print(input.shape)
                # print("outputs_f shape is {}".format(outputs_f[0].shape))
                outputs_f = []

        output = input, input[-1]
        return output
    @property
    def _flat_weights(self):
        return list(self._parameters.values())

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]
class MetaRNN(MetaRNNBase):
    def __init__(self, *args, **kwargs):
        super(MetaRNN, self).__init__('MetaRNN', *args, **kwargs)

class MetaLSTM(MetaRNNBase):
    def __init__(self, *args, **kwargs):
        super(MetaLSTM, self).__init__('MetaLSTM', *args, **kwargs)

