Yolov5 Optimization Doc
=================

Environment Setup
-----------
- `Pytorch 1.10 <https://pytorch.org/>`_  
- `Tensorflow (Test on 2.9.1) <https://www.tensorflow.org/install?hl=zh-tw>`_ 
- `NNI <https://nni.readthedocs.io/en/stable/index.html>`_ 
- `Fvcore <https://github.com/facebookresearch/fvcore>`_ 
- `Vision-toolbox <https://github.com/gau-nernst/vision-toolbox>`_

Quick start
-----------

.. note:: Pruning & One-shot NAS only works on Yolov5's backbone. We then map the new structured backbone back to construct a new detection model.  

Test one-shot nas on Yolov5's backbone

.. code-block:: bash

   $ python oneshot_nas.py 

Test pruning on Yolov5's backbone 

.. code-block:: bash

   $ python pruning.py 


Test multi-trial NAS on Yolov5's backbone 

.. code-block:: bash

   $ python hello_nas.py 

Test Knowledge Distillation on Yolov5 (Soft targets)

.. code-block:: bash

   $ python train.py --data coco.yaml --epochs 101 --weights "./checkpoint/student.pt" --t_weights "./checkpoint/teacher.pt"

Tutorial
-----------
This project managed several optimization methods on Yolov5, including:

* :doc:`Neural Architecture Search </nas/overview>`
* :doc:`Pruning </pruning/overview>`
* :doc:`Knowledge Distillation </kd/overview>`



Optimization API
-----------


Common Issues
-----------

