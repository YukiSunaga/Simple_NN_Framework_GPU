from np import *
from config import GPU
from models import Basic_Model, Hid_Model, Conv_Model, Conv_Model2, Conv_Model_WDDOBN, Conv_Model2_WDDOBN, Res_Model
from functions import *
from fashion_mnist import load_fashion_mnist
import pickle

(x_train, y_train), (x_test, y_test) = load_fashion_mnist(normalize=True, flatten=False, one_hot_label=True)
if GPU:
    x_train, x_test = to_gpu(x_train), to_gpu(x_test)
    y_train, y_test = to_gpu(y_train), to_gpu(y_test)

model = Basic_Model()
#model = pickle.load(open("result/201905111739/model.pkl", 'rb'))

model.fit(x_train, y_train, x_test, y_test)

model.save_all()
