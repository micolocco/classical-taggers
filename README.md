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

# Tagging decision
according to the convention used, B0, B+, B0s are given tagging decision = -1 (since they contain bbar), while B-, B0bar, B0sbar are given tagging decision = -1 (since they contain b).

If we want to get this information from the particles produced, we need to define the tagging decision as follow:

for OS Kaon, OS muon, OS electron in all decays + SS Pion and SS Kaon for charged decay only as: tagging_decision = charge of the track * -1
for SS Pion and SS Kaon in the case of B neutral decays as: tagging_decision = charge of the track

Jonas defined the tagging decision always as given by tagging_decision = charge of the track * -1 but actually this doesn't hold true for SS Pion and SS Kaon for neutral B if the general convention is that b -->-1 and bbar-->+1

also, the NN labels (ie the outputs that must be predicted) are:

0 (which is in origin -1 but then rescaled to 0) if the tag is correct
1 if the tag is wrong
Again, in order to maintain the convention and the definition of the label consistent, the label is defined as: 

7:57 PM







OS Kaon, OS muon, OS electron in all B decays + SS Pion and SS Kaon for charged B decays only as: label = B_TRUEID/abs(B_TRUEID) * B_Tr_T_Charge
for SS Pion and SS Kaon in the case of neutral B decays as: label = -B_TRUEID/abs(B_TRUEID) * B_Tr_T_Charge
