Desgin
=======

The development of deep learning (DL) algorithms and emerging DL models bring great challenges to underlying systems. Traditional DL systems, such as TensorFlow and PyTorch, have shown superior performance on various deep learning workloads due to their general characteristics and rich ecosystems. However, since the explosive growth of the scale of DL models and datasets, the distributed scalability is becoming the core competitiveness of DL systems. Although existing DL systems have provided some customized distributed training interfaces, they are still facing severe challenges and obstacles:

+ Functionality:
The supported communication architecture, parallel strategy and consistency protocal are limited.

+ Complexity:
The implementation of communication and compuation is highly coupled and hard to follow and optimize.

+ Usability:
The deployment of distributed training paradigms requires human expert knowledge for efficiency.

Besides, they are also suffering from the efficiency and scalability bottlenecks for large-scale distributed training. These observations motivate us to break the current system abstraction, make a novel design to handle all the above concerns and build a high-performance distributed DL system.

Hetu inherits the concept of data-flow graph (DFG) from existing deep learning frameworks, with operations as vertices and data dependencies as edges. The operation vertices not only represent computation kernels, but also consist of communication operators. Moreover, we provide a three-level representation of the DFG to describe and optimize distributed training programs.