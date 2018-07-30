import os
model_dir = './checkpoints/crnn/'
#model_dir = '/Users/simon/Desktop/OCR/CRNN_Tensorflow-master/model/shadownet/'
from tensorflow.python import pywrap_tensorflow
#checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
checkpoint_path = os.path.join(model_dir, "crnn_2018-07-20-17-19-39.ckpt-9")
#checkpoint_path = os.path.join(model_dir,"shadownet_2017-10-17-11-47-46.ckpt-199999")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    #print(reader.get_tensor(key))
