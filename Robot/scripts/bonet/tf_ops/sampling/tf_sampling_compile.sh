#/bin/bash
/usr/local/cuda/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# py27 + TF1.2
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow_core/include -I /usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# py27 + TF1.4
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow_core/include -I /usr/local/cuda/include -I /usr/local/lib/python2.7/dist-packages/tensorflow_core/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ -L /usr/local/lib/python2.7/dist-packages/tensorflow_core/ -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# py3.6 + TF1.2
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /home/bo/miniconda2/envs/py36tf12/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# py3.6 + TF1.4
#g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /home/bo/miniconda2/envs/py36tf14/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /home/bo/miniconda2/envs/py36tf14/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/home/bo/miniconda2/envs/py36tf14/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
