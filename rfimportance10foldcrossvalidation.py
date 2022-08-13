# coding = utf-8

from email import header
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RandomForest
from matplotlib.pyplot import MultipleLocator
from sklearn import utils
from sklearn import metrics
from sklearn import model_selection

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
ax=plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(40))


class RF:
    def __init__(self, test_train_rate=0.3):
        self.model = RandomForest(random_state=0)
        self.features = []
        self.feature_count = 0
        self.data = None
        self.label = ""
        self.test_train_rate = test_train_rate

    def read_file(self, file_name, label):
        df = pd.read_csv(file_name, header=0)
        headers = df.columns.tolist()
        if label not in headers:
            raise ValueError("所设定的标签在数据文件中不存在")

        headers.remove(label)
        self.features = headers
        self.feature_count = len(headers)
        self.label = label

        self.data = df

    def get_train_and_test_data(self):
        self.data = utils.shuffle(self.data, random_state=0)
        size = len(self.data)
        train_size = int((1 - self.test_train_rate) * size)

        train_x = self.data.loc[:train_size, self.features]
        train_y = self.data.loc[:train_size, self.label]

        test_x = self.data.loc[train_size:, self.features]
        test_y = self.data.loc[train_size:, self.label]

        return train_x, train_y, test_x, test_y

    def train(self, x, y):
        self.model.fit(X=x, y=y)

    def __sort(self, item):
        return item[1]

    def importance_sort(self, features):
        importance = list(self.model.feature_importances_)
        feature_importance = []

        for i in range(len(importance)):
            feature_importance.append((features[i], importance[i]))

        feature_importance.sort(key=self.__sort, reverse=True)
        return feature_importance

    def delete_feature_from_data(self, train_x, test_x, features):
        return train_x.loc[:, features], test_x.loc[:, features]

    def draw(self, x, y):
        # 设置画布大小
        plt.figure(figsize=(12, 8))
        # 绘图，设置坐标，width是直方块的宽度，facecolor是直方块的颜色
        rects = plt.bar(x, y, width=0.5, facecolor='black')
        plt.xticks(rotation=90)
        # 设置数据标签位置
        for rect in rects:
            rect_x = rect.get_x()  # 得到的是直方块左边线的值
            rect_y = rect.get_height()  # 得到直方块的高
            plt.text(rect_x + 0.5 / 2, rect_y + 0.5, str(int(rect_y)), ha='center', size=10)  # ha用于水平对齐
        plt.show()

    def extract(self, train_x, train_y, test_x, test_y, kf=0):
        bad_features = []
        # 第一次用所有特征进行训练
        rounds = []
        acc, precision_over_all, sen, f1, auc = self._single_fit(train_x, train_y, test_x, test_y)
        feature_importance_over_all = self.importance_sort(self.features)
        print(f"使用所有特征训练，准确率{precision_over_all}")
        print(f"acc: {acc}")
        print(f"sen: {sen}")
        print(f"f1: {f1}")
        print(f"auc: {auc}")
        print("开始特征消除")

        good_features = [f[0] for f in feature_importance_over_all]
        iter_round = 1
        tmp_precision = precision_over_all
        while True:
            kill = good_features.pop()
            print("-"*20, f"第{iter_round}轮", "-"*20)
            print(f"剔除特征  {kill}")
            # 剔除后重新排序

            _train_x, _test_x = self.delete_feature_from_data(train_x, test_x, good_features)
            acc, round_precision, sen, f1, auc = self._single_fit(_train_x, train_y, _test_x, test_y)
            round_feature_importance = self.importance_sort(good_features)

            print(f"剔除后准确率: {round_precision}")
            print(f"剔除后acc: {acc}")
            print(f"剔除后sen: {sen}")
            print(f"剔除后f1: {f1}")
            print(f"剔除后auc: {auc}")

            print(f"较之前增长了: {round_precision - tmp_precision}")
            print(f"剔除后剩余特征: {len(round_feature_importance)}")
            if iter_round == 110:
                x = [i[0] for i in round_feature_importance]
                y = [i[1] for i in round_feature_importance]
                self.draw(x, y)

            rounds.append([round_precision, good_features])
            tmp_precision = round_precision
            good_features = [f[0] for f in round_feature_importance]
            if iter_round < len(self.features) - 1:
                iter_round += 1
                if round_precision <= 0.6:
                    break
            else:
                break
        self.write_file(rounds, kf)
        x = []
        y = []
        for i in range(len(rounds)):
            x.append(self.feature_count - (i + 1))
            y.append(rounds[i][0])

        plt.plot(x, y, color='black', linewidth=1.0)
        plt.xlabel("变量个数")
        plt.ylabel("准确率")
        plt.savefig(f'{kf}.png')
        print("结束")

    def write_file(self, rounds, kf):
        with open(f"rounds{kf}.txt", "w", encoding="utf-8") as f:
            lines = []
            for i in range(len(rounds)):
                precision, features = rounds[i]
                s = f"第{i+1}轮迭代，准确率{precision}, 剔除了{[i for i in self.features if i not in features]}, 使用的特征{features}\n"
                lines.append(s)
            f.writelines(lines)

    def _extract_log(self, _precision, precision, feature_used, feature_index):
        print("-"*20 + f"第{feature_index}轮迭代" + "-"*20)
        print(f"上一轮准确率: {_precision}")
        print(f"此轮准确率 : {precision}")
        print(f"提升了: {precision - _precision}")

        print("此轮选用的特征:")
        for f in feature_used:
            print(f"   {f}")

    def _single_fit(self, train_x, train_y, test_x, test_y):
        self.model.fit(train_x, train_y)
        pre_y = self.model.predict(test_x)
        acc = metrics.accuracy_score(test_y, pre_y)
        precision = metrics.precision_score(test_y, pre_y)
        sen = metrics.recall_score(test_y, pre_y)
        f1 = metrics.f1_score(test_y, pre_y)
        auc = metrics.roc_auc_score(test_y, pre_y)
        return acc, precision, sen, f1, auc

    def measure(self, real, pre):
        real = list(real)
        pre = list(pre)
        error_count = 0
        size = len(real)
        for i in range(size):
            r = real[i]
            p = pre[i]
            if r != p:
                error_count += 1
        return (size - error_count) / size


# if __name__ == "__main__":
#     rf = RF()
#     rf.read_file(file_name="DTdatanotcfa.csv", label="LABEL")
#     trainX, trainY, testX, testY = rf.get_train_and_test_data()
#     rf.extract(trainX, trainY, testX, testY)
# 上述代码在无交叉验证时启用

if __name__ == '__main__':
    file = pd.read_csv("DTdatanotcfa.csv")
    label = "LABEL"
    y = file[label]
    X = file.drop(labels=[label], axis=1)
    headers = X.columns.to_list()
    kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
    i = 0
    for train_idx, test_idx in kf.split(X):
        x_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        print(f"=================交叉验证第{i+1}轮====================")
        i += 1
        model = RF()
        model.feature_count = len(headers)
        model.features = headers
        model.extract(x_train, y_train, x_test, y_test)
