# Распределенное глубокое обучение с параметрическим сервером
### Модели - EfficientNet, LeNet

### На сервере
```
pip install -r requirements.txt
ray start --head --resources='{"ps" : 1}'
```

### На вычислительных узлах
```
pip install -r requirements.txt
ray start --address='<То, что написано в консоли при запуске команд на сервере>' --resources='{"worker" : <количество возможных одновременно запущенных процессов на узле>}'
```

### После вышеперечисленных действий (на сервере)
```
python main.py -m=efficientneet -d=imagenet
```
