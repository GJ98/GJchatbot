import os
import platform
import torch

root_dir = os.path.abspath(os.curdir)

_ = '\\' if platform.system() == 'Windows' else '/'
if root_dir[len(root_dir) - 1] != _: root_dir += _

PATH = {
    'root_dir': root_dir,  
    'delimeter': _,  
    'raw_data_dir': root_dir + "data{_}raw{_}".format(_=_),  
    'ood_data_dir': root_dir + "data{_}ood{_}".format(_=_),  
    'intent_data_dir': root_dir + "data{_}intent_data.csv".format(_=_),  
    'entity_data_dir': root_dir + "data{_}entity_data.csv".format(_=_),  
    'model_dir': root_dir + "saved{_}".format(_=_),  
}
