.. rst-class:: hide-header

.. rst-class:: hide-header

.. image:: _static/iprdlogo.gif
   :scale: 15%
   :alt: IPRD Group
   :align: right


*********************
Welcome to RDT-Reader
*********************

.. toctree::
   :maxdepth: 2


Quickstart
----------

    1. Enable git LFS so that the models can be downloaded. Refer_

    .. _Refer: https://help.github.com/en/articles/installing-git-large-file-storage

    2. Clone this repo into <rdt-reader> .  

    .. code-block:: 

        git clone --recurse-submodules https://github.com/DigitalHealthIntegration/rdt-reader.git

    3. Move the tensorflow-yolov3-models/models sub folder in tensorflow-yolov3/

    4. For data annotation you can use any tool that you like. We used VOTT_ version 1.7.2

    5. Install anaconda to recreate the python enviroment.

    .. _VOTT: https://github.com/Microsoft/VoTT/releases

Data Format
------------

    Two data formats are used, the first one listed below is to train the object detection model and the second for the red and blue line detection.

    1. Example_

    .. _Example: https://github.com/kashyapj2793/tensorflow-yolov3/blob/master/data/dataset/rdt_train.txt
        Make sure you change the path in this file before training.

    2. You can find an example of the annotations_ in the labels folder, each file contains annotations for an image with the same base file name in this folder_.


    .. _annotations: https://github.com/DigitalHealthIntegration/rdt-reader/tree/master/dataset/labels

    .. _folder: https://github.com/DigitalHealthIntegration/rdt-reader/tree/master/dataset/images_yolo

Set-up python enviroment
-----------------
    .. code-block:: 

        conda create --name <envname> --file requirements.txt python=3.6
        conda activate <envname>

Using service with pretrained models
------------------------------------

    .. code-block:: 

        python flasker.py

    A local flask server will be setup running on localhost:9000.

    You can hit this server as described in the documentation here_.

    .. _here: https://drive.google.com/open?id=1Tbz2k5p9v9xEIhrNy9WP4w5mPTyPjVOPRIxeHn87r74

    There is a saved postman collection (audere_local.postman_collection) in the root directory which can be used to test the API with Postman. Please change the image file to a file that resides on your system.

Train Object recognition model
------------------------------

    For detailed instructions please follow the README_ . Edit the config_ file first and set the absolute paths correctly.

    .. _README: https://github.com/kashyapj2793/tensorflow-yolov3/blob/master/README.md

    .. _config: https://github.com/kashyapj2793/tensorflow-yolov3/blob/master/core/config.py


    .. code-block:: 

        cd tensorflow
        python train.py

    You will find models being saved in the checkpoint_ directory. 

    .. _checkpoint : https://github.com/kashyapj2793/tensorflow-yolov3/tree/master/checkpoint

Freeze Object recognition model for inference
---------------------------------------------

    Run the script_ . Please change the paths to select the newly trained model and new export directory.

    .. _script : https://github.com/kashyapj2793/tensorflow-yolov3/blob/master/freeze_yolo_tf.py


Train red-line detection model
------------------------------

    .. code-block:: 

        cd ..
        python train_blue_red.py

    You can change the name and location of the saved model from within the script. 


Freeze red-line detection model for inference
---------------------------------------------

    Use the script :ref:`freezeline` . Please change the paths to select the newly trained model and new export directory.

Indices and tables
==================
 
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
