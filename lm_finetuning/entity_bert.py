from pytorch_pretrained_bert.modeling import BertForPreTraining

class EntityBert(BertForPreTraining):

    def __init__(self, config):
        super(EntityBert, self).__init__(config)
