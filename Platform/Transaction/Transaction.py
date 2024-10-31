import hashlib
class transaction(object):
    def __init__(self, runtype, publisher, timespan, model, approval_list):
        self.publisher = publisher  # 发布交易的节点编号
        self.type = runtype  # 交易类型，1为分区内，2为跨区
        self.timespan = timespan  # 交易创建的时间
        self.model = model
        self.approved_time = []  # 交易被批准的时间
        self.approved = 0  # 交易被批准次数
        self.approval = approval_list  # 指向（批准）的交易
        self.acc = 0.0  # 模型精度
        self.acc_choice = 0.0
        self.loss = 0.0  # 模型损失
        self.iterations = 0
        self.name = self.gen_name()
        self.hash = self.gen_hash()

    def gen_name(self):  # 生成交易名称
        # trans_name = str(self.type) + '_' + str(self.publisher) + '_' + str(self.timespan)
        trans_name = str(self.publisher) + '_' + str(self.timespan)
        return trans_name

    def gen_hash(self):  # 生成交易哈希
        trans_hash = self.calculate_hash256(self.name)
        return trans_hash

    def calculate_hash256(self, s):
        sha256_hash = hashlib.sha256()
        sha256_hash.update(s.encode('utf-8'))
        return sha256_hash.hexdigest()