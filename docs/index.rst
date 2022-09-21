Yolov5 Optimization Doc
=================

Environment Setup
-----------
- `Pytorch 1.10 <https://pytorch.org/>`_  
- `Tensorflow (Test on 2.9.1) <https://www.tensorflow.org/install?hl=zh-tw>`_ 
- `NNI <https://nni.readthedocs.io/en/stable/index.html>`_ 
- `Fvcore <https://github.com/facebookresearch/fvcore>`_ 
- `Vision-toolbox <https://github.com/gau-nernst/vision-toolbox>`_

.. note:: Testing ... 


Overview
-----------
This project managed several optimization methods on Yolov5, including:

* :doc:`Neural Architecture Search </nas/overview>`
* :doc:`Pruning </pruning/overview>`
* :doc:`Knowledge Distillation </kd/overview>`

Quick start
-----------

Test one-shot nas on Yolov5 

.. code-block:: bash

   $ python oneshot_nas.py 

Optimization API
-----------


