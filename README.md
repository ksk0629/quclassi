弊社の社内研究にて、量子機械学習の研究の一環として作成したソースコードを公開する。本ソースコードは社内の別の開発者の許可をいただいて公開している。

なお、ソースの共有が目的のため、結果については記載しない。またメンテナンスも原則行わない。

# QuClassi
## 環境
- python 3.8.12
- module
    - mlflow 1.24.0
    - qiskit 0.31.0
    - scikit-learn 1.0.1 

## クイックスタート
以下を実行する。

```
python ./src/[objective_script]
```

なお、`objective_script`については以下の表のスクリプト名部分のもののどれかが入る。
| スクリプト名 | クラス数 | データの次元 |
| --- | --- | --- |
| `breast_cancer_model.py` | 2 | 30 |
| `iris_model.py` | 3 | 4 |
| `wine_model.py` | 3 | 13 |