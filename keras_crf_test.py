#用keras实现计算loss的单独CRF层
from keras.layers import Layer
import keras.backend as K

class CRF(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    """
    def __init__(self, ignore_last_label=False, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs) #用super继承Layer类
    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        #初始化转移矩阵
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        states = K.expand_dims(states[0], 2) # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0) # (1, output_dim, output_dim)
        output = K.logsumexp(states+trans, 1) # (batch_size, output_dim)
        return output+inputs, [output+inputs]
    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        """
        这个所谓的目标路径的相对概率就是目标函数中后面一项（被减的部分），
        根据CRF的原理，包括两个部分：
        状态转移概率（两个相邻状态之间的关系）和状态概率（观测节点和状态节点之间的关系）
        我们的目标就是最小化loss，最大化这个目标路径的相对概率。
        """
        #point_score就是在状态概率上的得分，用前面的LSTM等模型预测的标签序列与正确的标签序列进行点乘，
        # 只关心每个标签是否预测正确，预测标签是由前面的LSTM模型得到
        point_score = K.sum(K.sum(inputs*labels, 2), 1, keepdims=True) #逐标签得分
        #trans_score就是在状态转移概率上的得分，不关心每个标签是否预测正确，
        # 只关心标签和标签之间的转移本身是否合理，CRF的任务
        labels1 = K.expand_dims(labels[:, :-1], 3)  #labels是正确标签的one-hot序列
        labels2 = K.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2 # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        print("labels", labels.shape)
        #经过点乘，labels记录了标签的转移特征
        print("trans_befor", self.trans.shape)
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        print("trans_next", trans.shape, trans)
        print("trans*labels:", trans*labels)
        trans_score = K.sum(K.sum(trans*labels, [2,3]), 1, keepdims=True)
        print("trans_score", trans_score)
        return point_score+trans_score # 两部分得分之和
    def call(self, inputs): # CRF本身不改变输出，它只是一个loss
        return inputs
    def loss(self, y_true, y_pred): # 目标y_pred需要是one hot形式
        print("CRF正在计算loss...")
        mask = 1-y_true[:,1:,-1] if self.ignore_last_label else None
        # print(y_true, y_pred)
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        init_states = [y_pred[:,0]] #初始状态
        # 通过rnn用递归地法计算规范化因子Z向量（未取对数之前的值），其中log_norm_step参数是一个函数
        log_norm,_,_ = K.rnn(self.log_norm_step, y_pred[:,1:], init_states, mask=mask) # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True) #对规范化因子取对数
        path_score = self.path_score(y_pred, y_true) # 计算分子（对数）
        #log_norm和path_score中都包括了slef.trans：
        #trans保存了各个标签之间的转移概率，不管是log_norm还是path_score都要用到trans
        # 进行计算，该CRF前面的模型对每个节点的输出进行预测，然后该CRF层引入转移概率矩阵参数trans，
        # 对预测序列之间的关系进行限定（若出现不合理的标签转移则增加loss，之前的LSTM模型无法做到这一点）
        return log_norm - path_score # 即log(分子/分母)
    def accuracy(self, y_true, y_pred): # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        print("CRF正在计算accuracy...")
        mask = 1-y_true[:,:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal*mask) / K.sum(mask)