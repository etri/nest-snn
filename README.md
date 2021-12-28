## About the Project

The repository contains some examples of pre-trained SNN (Spiking Neural Network) models. 

The models were trained using the [MM-BP](https://github.com/jinyyy666/mm-bp-snn) training algorithm, and the models run on the [CARLsim4](https://github.com/UCI-CARL/CARLsim4) SNN simulator. 

The models aim to perform image classification or speech recognition. For the image classification, the [MNIST](http://yann.lecun.com/exdb/mnist/) or [N-MNIST](http://www.garrickorchard.com/datasets/n-mnist) dataset is used. For the speech recognition, the [TI46](https://catalog.ldc.upenn.edu/LDC93S9) dataset is used.

## Getting Started

The project has been tested under the Linux Ubuntu 16.04 LTS with CUDA 9.0.

### Prerequisities

1. Linux (We have tested under ubuntu 16.04)
2. CUDA 9.0 (Using other CUDA verions may not work)

### Getting Started

1. Download [CARLsim4](https://github.com/UCI-CARL/CARLsim4)

2. Download this project
    ```sh
    git clone git@github.com:etri/nest-snn.git
    ```
3. Patch modified CARLsim4 source code
    ```sh
    cd CARLsim4
    patch -p1 < carlsim.patch
    ```
    (Note) The neuron and synapse models assumed in MM-BP differ from those provided by CARLsim4. 
    So, we additionally implemented the neuron and synapse models in CARLsim4. 
    And we provide the additional implementation in the form of patch file(carlsim.patch) upon request.
    Please contact us if you would like to use the patch file.
    
4. Install [CARLsim4](https://github.com/UCI-CARL/CARLsim4) following the CARLsim4 installation process.
    
### Running Examples 

1. MNIST
    ```
    cd snn_models/MNIST_trained
    make
    ./trained_mnist 
    ```
2. N-MNIST
    ```
    cd snn_models/N-MNIST_trained
    make
    ./trained_nmnist 
    ```
3. TI46
    ```
    cd snn_models/TI46_trained
    make
    ./trained_ti46
    ```

### Get Dataset
1. MNIST dataset 

    Get the [MNIST dataset](http://yann.lecun.com/exdb/mnist). 
    Place the dataset in the snn_models/MNIST_trained/mnist.
    
2. N-MNIST dataset

    Get the [N-MNIST dataset](http://www.garrickorchard.com/datasets/n-mnist).
    Then encode data following the instructions provided by [MM-BP github page](https://github.com/jinyyy666/mm-bp-snn).
    
    We provide some input samples(snn_models/N-MNIST_trained/sample_inputs) for the test. 
3. TI46 dataset

    TI 46 dataset is not free, and the source code used for encoding has not been opened.
    Thus we do not provide input files. If you contact us via email, we can guide you on how to obtain the dataset and how to encode it.
    
<!-- LICENSE -->
## License
This project is licensed under [Apache 2.0 License](LICENSE).

<!-- CONTACT -->
## Contact
PAK,EUNJI - pakeunji@etri.re.kr

Project Link: [https://github.com/pakeunji/nest-snn.git](https://github.com/pakeunji/nest-snn.git)



