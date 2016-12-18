# classifier-comparison
Provide a visual way to compare several classification algorithms

Examples of use

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
compare_logistic_model(url, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

![alt tag](https://github.com/agnesmm/classifier-comparison/blob/master/iris.png)


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data'
compare_logistic_model(url, ['No-use', 'Long-term', 'Short-term'])


![alt tag](https://github.com/agnesmm/classifier-comparison/blob/master/contraceptive_method.png)


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
compare_logistic_model(url, ['class 0', 'class 1'])

![alt tag](https://github.com/agnesmm/classifier-comparison/blob/master/banknote_authentication.png)

