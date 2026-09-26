# Gradient — sıfırdan çok katmanlı sinir ağı (gradyan inişi)

NumPy ile yazılmış, istenen sayıda gizli katmanı olan bir ileri beslemeli sinir ağı. Eğitim, geri yayılım ve gradyan inişiyle yapılır. `dnn.py` ağı (`neuralNetwork`: `train`, `query`) içerir; `inuse.py` 5 girdi, [4, 10, 2] gizli ve 3 çıktılı küçük bir kullanım örneğidir.

```bash
pip install numpy
python inuse.py
```

Bu depo çalışmanın ilk sürümüdür. Belgelenmiş, örnekli ve güncel sürüm: [Proaiml/DNN_GD](https://github.com/Proaiml/DNN_GD).

## Test

```bash
pip install pytest
python -m pytest tests -q
```

Duman testleri yalnızca CPU kullanır ve birkaç saniyede biter.
