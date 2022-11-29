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

.. note::  Some methods only works on Yolov5's backbone (ex: NAS). We then map the new structured backbone back to construct a new detection model.  

Test one-shot NAS on Yolov5's backbone

.. code-block:: bash

   $ python oneshot_nas.py 


Test multi-trial NAS on Yolov5's backbone 

.. code-block:: bash

   $ python hello_nas.py 

Test pruning on Yolov5's backbone 

.. code-block:: bash

   $ python pruning.py 

Test Iterative pruning on Yolov5

.. code-block:: bash

   $ bash iterative_pruning.sh 

Test Knowledge Distillation on Yolov5 (Soft targets)

.. code-block:: bash

   $ python train.py --data coco.yaml --epochs 101 --weights "./checkpoint/student.pt" --t_weights "./checkpoint/teacher.pt"


Test Feature Distillation on Yolov5

.. code-block:: bash

   $ python train.py --data coco.yaml --epochs 101 --weights "./checkpoint/multi_nas_yolov5s.pt" --ft_weights "./checkpoint/yolov5m.pt" 

Tutorial
-----------
This project managed several optimization methods on Yolov5, including:

* :doc:`Neural Architecture Search </nas/overview>`
* :doc:`Pruning </pruning/overview>`
* :doc:`Knowledge Distillation </kd/overview>`



Optimization API
-----------
* :doc:`Optimization API </API/overview>`

Common Issues
-----------
* :doc:`Common Issues </Issues/overview>`


Experiment Results (only 100 epochs for each model)
-----------
.. image:: ./Result.png 
   
