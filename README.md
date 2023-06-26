# RNX2VTX

国土地理院GEONET等で公開されているGNSS衛星RINEXデータをTECデータに変換するためのプログラムです。以下の処理を行えます。
1. RINEXデータから、バイアスを含んだTECデータを作成
2. 衛星・受信局ごとのコード間バイアス(DCB)を推定
3. 1. 2.のデータからバイアスを含まないTECデータを推定
  
2. のDCB推定は以下の論文(1)で示されているアルゴリズムを用いています。ただし、Slant Factorとして1/cosZではなく、S1/S0を採用しています。ここで,
S1/S0は論文(2)で定義されたS1,S0を採用しています。

(1) Ma, G. and Maruyama, T.: Derivation of TEC and estimation of instrumental biases from GEONET in Japan, Ann. Geophys., 21, 2083–2093, https://doi.org/10.5194/angeo-21-2083-2003, 2003.
(2) Otsuka, Y., Ogawa, T., Saito, A. et al. A new technique for mapping of total electron content using GPS network in Japan. Earth Planet Sp 54, 63–70 (2002). https://doi.org/10.1186/BF03352422

・使用方法
(1) 本プログラムおよびTECデータに変換したい日付のRINEXデータを用意してください。
(2) 本プログラム中で日付、RINEXバージョン、入力ファイルおよび出力ファイルのパス、各種パラメータ(後述)を設定してください。
(3) コードを実行してください。

・パラメータについて
本プログラムで指定可能なパラメータは以下の通りです。
・読み取るRINEXデータのコード(L1C,C2Xなど)
・有効なデータとみなされる一連のデータの長さの下限（デフォルト:60エポック）
・欠損としても一連のデータとみなされるデータの長さの上限(デフォルト:2エポック)
・処理2において、VTEC値を一定とみなす時刻(デフォルト:30エポック)、範囲(デフォルト:1度x1度)
これらのパラメータは変更することが可能です。変更する場合はプログラムの該当部分を変更してください。


・想定している言語
Python 3.10.7

・必要なライブラリ
・NumPy 1.22.4
・Scipy 1.9.2
・math 
・datetime
・os
・glob
