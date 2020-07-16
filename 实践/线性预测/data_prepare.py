feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'convert'
]
feature_num = len(feature_names)
data = numpy.fromfile(filename, sep=' ') # Read primary data from file
data = data.reshape(data.shape[0] // feature_num, feature_num)
maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0)/data.shape[0]

for i in six.moves.range(feature_num-1):
   data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i]) # six.moves is compatible to python2 and python3

ratio = 0.8 # distribution ratio of train dataset and verification dataset
offset = int(data.shape[0]\*ratio)
train_data = data[:offset]
test_data = data[offset:]

def reader_creator(train_data):  
    def reader():  
        for d in train_data:  
            yield d[:-1], d[-1:]  
    return reader

train_reader = paddle.batch(
    paddle.reader.shuffle(
        reader_creator(train_data), buf_size=500),
        batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        reader_creator(test_data), buf_size=500),
        batch_size=BATCH_SIZE)