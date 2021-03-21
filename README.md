# is_the_oreder_a_keras
ご注文はKerasですか？

流れとしては、小規模データを使い、それによって簡易的なニューラルネットワークを形成、それによって大規模データを自動的に収集する。
その大規模データを用いて学習する、という流れになります。

以下、各プログラムについて簡単な説明となります。

<h3>get_image.py</h3>
初めに動画から画像を収集する

<h3>call_me_keras.py</h3>
小規模データを用いて学習。データ数が圧倒的に足りないため、batch_sizeは1（単純な SGD）に。<br>
この時点でも、ある程度の認識はできるようになります。

<h3>extract_bigdata.py</h3>
前までに作ったニューラルネットワークを用いて大規模データを収集する

<h3>extract_testdata.py</h3>
テスト検証用データを生成する

<h3>image_generator.py</h3>
大規模データのかさ増しをする

<h3>create_trainimage.py</h3>
訓練データとして複数のフォルダにある画像を１つのフォルダにまとめる

<h3>call_me_keras.py</h3>
大規模データの学習。

<h3>no-poi.py</h3>
大規模データの学習後、それを学習させ、枠付けを行った動画を出力
