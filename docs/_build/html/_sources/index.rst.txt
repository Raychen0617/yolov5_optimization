YOLOv5 Optimization Doc
=================

Environment Setup
-----------
- `Pytorch 1.10 <https://pytorch.org/>`_  
- `Tensorflow (Test on 2.9.1) <https://www.tensorflow.org/install?hl=zh-tw>`_ 
- `NNI <https://nni.readthedocs.io/en/stable/index.html>`_ 
- `Fvcore <https://github.com/facebookresearch/fvcore>`_ 
- `Vision-toolbox <https://github.com/gau-nernst/vision-toolbox>`_
- `Cuda (Test on 11.6.55) <https://developer.nvidia.com/cuda-toolkit-archive>`_
Quick start
-----------

.. note::  Some methods only works on YOLOv5's backbone (ex: NAS). We then map the new structured backbone back to construct a new detection model.  

Test one-shot NAS on YOLOv5's backbone

.. code-block:: bash

   $ python oneshot_nas.py 


Test multi-trial NAS on YOLOv5's backbone 

.. code-block:: bash

   $ python hello_nas.py 

Test pruning on YOLOv5's backbone 

.. code-block:: bash

   $ python pruning.py 

Test Iterative pruning on YOLOv5

.. code-block:: bash

   $ bash iterative_pruning.sh 

Test Knowledge Distillation on YOLOv5 (Soft targets)

.. code-block:: bash

   $ python train.py --data coco.yaml --epochs 101 --weights "./checkpoint/student.pt" --t_weights "./checkpoint/teacher.pt"


Test Feature Distillation on YOLOv5

.. code-block:: bash

   $ python train.py --data coco.yaml --epochs 101 --weights "./checkpoint/student.pt" --ft_weights "./checkpoint/teacher.pt"

Optimization Tutorial
-----------
This project managed several optimization methods on YOLOv5, including:

* :doc:`Neural Architecture Search </nas/overview>`
* :doc:`Pruning </pruning/overview>`
* :doc:`Knowledge Distillation </kd/overview>`



Optimization API
-----------
* :doc:`Optimization API </API/overview>`

Common Issues
-----------
* :doc:`Common Issues </Issues/overview>`


Experiment Results 
-----------
.. note:: - Baseline is the result of YOLOv5s train 100 epochs from scratch.
          - Every model in the result is an optimized YOLOV5s and is trained for 100 epochs from scratch.
          - NAS v1 and v2 (ex: DARTs_v1, DARTs_v2) differs from search space. v2 has larger search space than v1.
          

.. image:: ./Final_Result.png 
   
