# 安装scikit-learn
from sklearn import svm

# 二分类问题
# 训练样本特征
X = [[0, 0], [1, 1]]
# 训练样本类别标签
Y = [0, 1]
# 创建向量机
clf = svm.SVC()
# 训练
clf.fit(X, Y)
# 测试数据样本
test = [[2, 2]]
# 预测
result = clf.predict(test)
print(result)

# 多分类问题
X = [[0], [1], [2], [3], [4]]
Y = [0, 1, 2, 3, 4]
test = [[6]]
# 选择一对一策略
clf1 = svm.SVC(decision_function_shape='ovo')
clf1.fit(X, Y)
# 查看投票函数
dec = clf1.decision_function(test)
# 查看筛选函数的大小，可以看到是10， 是因为ovo策略会设计5*4/2=10个分类器，然后找出概率最大的
print(dec.shape)
# 预测
result = clf1.predict(test)
print(result)
