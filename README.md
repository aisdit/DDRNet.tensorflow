## requirements
 - python: 3.6
 - tensorflow-gpu 2.6.2
 - opencv-python
 - yacs

## folder structure
    📦DDRNet.tensorflow
    ┣ 📂config                              # config files
    ┃ ┗ 📂giant
    ┃ ┃ ┗ 📜ddrnet23_slim.yaml
    ┣ 📂data                                # test data
    ┃ ┗ 📜0.jpg
    ┣ 📂lib
    ┃ ┗ 📂config                            # config declaration
    ┃ ┃ ┣ 📜default.py
    ┃ ┃ ┗ 📜__init__.py
    ┃ ┗ 📜utils.py
    ┣ 📂models                              # pretrained models converted from pytorch
    ┃ ┣ 📂ddrnet23_slim_4class.pb
    ┃ ┗ 📂ddrnet23_slim_lab2_2class.pb
    ┣ 📂output
    ┣ 📜test_tf.py                          # segmentation inference