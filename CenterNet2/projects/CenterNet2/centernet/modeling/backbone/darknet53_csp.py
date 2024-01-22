from .common import *
from copy import deepcopy
import torch.nn as nn
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.meta_arch import rcnn

class CSP_Darknet(Backbone):


    def __init__(self,cgf_path, out_features, in_channels ,num_classes):
        super().__init__()


        self.num_classes = num_classes
        self. _out_features= out_features
        self._out_feature_channels ={'res3':256, 'res4': 512, 'res5': 1024}
        self._out_feature_strides= {'res3':8, 'res4': 16, 'res5': 32}

        self.yaml, self.ch= load_cfg(cgf_path, in_channels, num_classes)
        self.yaml= deepcopy(self.yaml)
        #print(f"The config: {self.yaml}")
        self.layers, self.save = parse_model(self.yaml, ch=[self.ch])
        self.model = nn.ModuleList(self.layers)
        initialize_weights(self)

        #print(f"The layers: {self.layers}")


    def forward(self, x):
        y = []  # outputs
        outputs={}
        rcnn.local_global_features = {}
        
        for idx, m in enumerate(self.model):

            #print("Layer Number: {} Module Type: {}".format(idx,type(m)))
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

            if idx in self.yaml['out_features']:
                outputs['res'+str(len(outputs)+3)]=x

            if idx == 21:
                rcnn.local_global_features['local_features']=x

            if idx == 25:
                rcnn.local_global_features['global_features']=x

        return outputs

# if __name__ == "__main__":
    
#     model= CSP_Darknet("models/yolov5l.yaml",3,10)

#     print(model)
#     input=torch.rand(4, 3, 768, 768)

#     outputs= model(input)

#     for x in outputs:
#         print(outputs[x].shape)




