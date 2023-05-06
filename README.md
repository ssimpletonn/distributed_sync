# Распределенное глубокое обучение с параметрическим сервером
# Модель - EfficientNet

### На сервере
```
pip install -r requirements.txt
ray start --head --resources='{"ps" : 1}'
```

### На вычыслительных узлах
```
pip install -r requirements.txt
ray start --address='<То, что написано в консоли при запуске команд на сервере' --resources='{"worker" : <количество возможных запущенных процессов на узле>}'
```

### После вышеперечисленных действий (на сервере)
```
python main.py
```
