#/bin/bash
g++ tf_grouping_g_cpu.cpp -o tf_grouping_g_cpu.o -c -O2 -fPIC

g++ -std=c++11 tf_grouping_cpu.cpp tf_grouping_g_cpu.o -o tf_grouping_cpu.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow_core/include -I /usr/local/lib/python2.7/dist-packages/tensorflow_core/include/external/nsync/public -L /usr/local/lib/python2.7/dist-packages/tensorflow_core/ -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
