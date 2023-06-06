# Classical taggers
Instructions for launching the following scripts:
- `pyTorchTraining.py`: script used for training the Neural Network. Usage:
    `python pyTorchTraining.py <DecayType> <TaggerType>` (ex. `python pyTorchTraining.py Bu2JpsiK OSKaon`)
- `pipeline.py`: script that modifies the NTuples created previously (one for each decay). TO BE REPRODUCED WITH UPSTREAM TRACKS.
## PyTorch C++ interface
- The code implementation for loading PyTorch models into C++ refers to https://pytorch.org/tutorials/advanced/cpp_export.html.
- Please note that you need `libtorch` for loading PyTorch models into C++. Follow instructions here https://pytorch.org/ to download it.
- In order to have a compatible `gcc` version, you need to do:

    `source /cvmfs/sft.cern.ch/lcg/contrib/gcc/8.2.0/x86_64-centos7/setup.sh`

C++ commands to run :
1) make sure to have the `Cimplementation/build` folder otherwise inside `Cimplementation` make it as:

    `mkdir build`
2) make sure the `build` folder is empty
3) `cd build`
4) `cmake -DCMAKE_PREFIX_PATH=<your path to libtorch library> ..` 
5) `make` (this can take a bit)
6) `./C_modelLoader <model_path>`
