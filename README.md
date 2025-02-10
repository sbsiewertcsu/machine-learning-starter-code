# machine-learning-starter-code
Starter code in C++ and Python for Machine Learning

For Python on a public machine, create an environment for your account
python -m venv ~/public_html/csci581/code

After creation, be sure to enter that Python environment:
source bin/activate

Pip install packages needed:
---------------------------
E.g., python3 -m pip install --upgrade matplotlib
python3 -m pip install --upgrade tensorflow -- this one is big, so check space wtih "quota -s" first
python3 -m pip install --upgrade jupyterlab -- this one is big too

(code) sbsiewert@ecc-linux2:~/public_html/csci581/code/pycv_demo$ python3 -m pip install --upgrade opencv-python
Collecting opencv-python
  Downloading opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (63.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.0/63.0 MB 20.6 MB/s eta 0:00:00
Requirement already satisfied: numpy>=1.21.2 in /var/www/sbsiewert/csci581/code/lib/python3.10/site-packages (from opencv-python) (2.0.2)
Installing collected packages: opencv-python
Successfully installed opencv-python-4.11.0.86
(code) sbsiewert@ecc-linux2:~/public_html/csci581/code/pycv_demo$


Run Python examples:
-------------------

Simple OpenCV Python program:
python3 demo.py

Simple TensorFlow CNN program:

(code) sbsiewert@ecc-linux2:~/public_html/csci581/code/testtf$ python3 basic.py
Installing dependencies for Colab environment
2025-02-05 11:40:26.488569: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers o                                                       n your machine, GPU will not be used.
2025-02-05 11:40:26.496925: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers o                                                       n your machine, GPU will not be used.
2025-02-05 11:40:26.516070: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register                                                        cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1738784426.547229 1030433 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to regist                                                       er factory for plugin cuDNN when one has already been registered
E0000 00:00:1738784426.556006 1030433 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to regi                                                       ster factory for plugin cuBLAS when one has already been registered
2025-02-05 11:40:26.590161: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is opt                                                       imized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate c                                                       ompiler flags.

TensorFlow version: 2.18.0

Getting Fashion MNIST dataset

train_images.shape: (60000, 28, 28, 1), of float64
test_images.shape: (10000, 28, 28, 1), of float64

Creating Keras model
/user/home/sbsiewert/public_html/csci581/code/lib/python3.10/site-packages/keras/src/layers/convolutional/base                                                       _conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential                                                        models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
2025-02-05 11:40:34.802332: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to c                                                       uInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ Conv1 (Conv2D)                       │ (None, 13, 13, 8)           │              80 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 1352)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ Dense (Dense)                        │ (None, 10)                  │          13,530 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 13,610 (53.16 KB)
 Trainable params: 13,610 (53.16 KB)
 Non-trainable params: 0 (0.00 B)

 model compiled!
2025-02-05 11:40:35.447122: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 188                                                       160000 exceeds 10% of free system memory.
Epoch 1/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - loss: 0.7170 - sparse_categorical_accuracy: 0.7572
Epoch 2/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 3ms/step - loss: 0.3872 - sparse_categorical_accuracy: 0.8646
Epoch 3/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 3ms/step - loss: 0.3454 - sparse_categorical_accuracy: 0.8774
Epoch 4/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 3ms/step - loss: 0.3227 - sparse_categorical_accuracy: 0.8845
Epoch 5/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 3ms/step - loss: 0.3099 - sparse_categorical_accuracy: 0.8881

 model trained!
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 0.3430 - sparse_categorical_accuracy: 0.8783

Test accuracy: 0.8758999705314636

 model tested!
export_path = /tmp/basic.keras


Saved model as /tmp/basic.keras

Saved model has been reloaded
(code) sbsiewert@ecc-linux2:~/public_html/csci581/code/testtf$

