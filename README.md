# Classical taggers
Instructions for launching the following scripts:
- `pyTorchTraining.py`: script used for training the Neural Network. Usage:
    `python pyTorchTraining.py <DecayType> <TaggerType>` (ex. `python pyTorchTraining.py Bu2JpsiK OSKaon`)
- `pipeline.py`: script that modifies the NTuples created previously (one for each decay). TO BE REPRODUCED WITH UPSTREAM TRACKS.

## Producing NTuples
sequence:
1) you need to produce the list of LNF from the `All_LFN.txt`:
`cd classical-taggers/Data/moore_scripts/LFNs/`
`python cuts_LFN.py <number_of_files> <decay>`
2) `cd ../SM_lines`
`python produce_lines.py <number_of_files> <decay>`
3) go back to the `classical-taggers` folder and launch `snakemake`:
`snakemake -n -j <number_of_jobs>`
ALternatively you can run Moore and DaVinci seperately with:
`<path_to_stack>/Moore/build.x86_64_v2-centos7-gcc12+detdesc-opt/run gaudirun.py Data/moore_scripts/SM_lines/<decay>/line_SnakeMake_<n>.py` 
`<path_to_stack>/DaVinci/build.x86_64_v2-centos7-gcc12+detdesc-opt/run lbexec Data/daVinci_scripts/Bu2JpsiK/Algs.py:main Data/daVinci_scripts/Bu2JpsiK/options_SM/options_<n>.yaml`
where `n` is the option file for the n-th LFN 
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

# Schema for Tagging Decision and NN label assignment 

w = not a tagging particle for that B decay as a signal
r = tagging particle for that B decay as a signal

   |     | $\pi^+$ | $\pi^-$ |$K^+$ | $K^-$ |
   | --- | --- | --- | ---|---|
$B^+$ | w   | r   |/|/|
$B^-$ | r   | w   | /|/|
$B^0$ |r    | w   |/|/|
$\bar{B^0}$ | w   | r   |/|/|
$B^0_s$ |/    | /   |r|w|
$\bar{B^0_s}$ | /  | /   |w|r|

## Tagging decision
**$Q_f$ = flavour charge.** 
* $Q_f$ = +1 $\rightarrow$ It's a particle
* $Q_f$ = -1 $\rightarrow$ It's an anti-particle

**$d$ = tagging decision d[$Q_c$(track), B species, tagger].** 
It's a function of the charge of the track $Q_c$, the $B$ decay species and the tagger type.
* $d = +1$ $\rightarrow$ the signal has a $\bar b$
* $d = -1$ $\rightarrow$ the signal has a $b$
```
    d = +1 * Qc(track) if Bs0 and SSK
    d = -1 * Qc(track) if Bd0 and SSK*
    d = +1 * Qc(track) if Bd0 and SSpi
    d = -1 * Qc(track) if Bd0 and SSp
    d = -1 * Qc(track) if B+ and any tagger
    d = -1 * Qc(track) if Bd0 and any OS tagger
    d = -1 * Qc(track) if Bs0 and any OS tagger
  ```  
$*B^0$(bbar, d) and $K^*$(dbar, s)->$K^-$(s, ubar)$\pi^+$(dbar, u). We can have SS $K$ taggers for $B^0$ as well but in this situation it's mostly used the SS $\pi$ as a tagging particle.

## Label
**label = output of the Neural Network** 
It's a fucntion of the tagging decision $d$ and of the $B$ flavour charge $Q_f$.
It defines if the tag is correct (label = +1) or wrong (label = -1*) (ie the tagging decision is correct when it has the same sign of the $B$ flavour charge).

    label(d, Qf(B)) = d * Qf(B) = (+-1) * Qc(track) * Qf(B)    

where Qf(B) = B_TRUEID/abs(B_TRUEID)

* Assuming $B^+, SS\pi$
    * $\pi^+ \rightarrow d = -1, \text{label} = -1$ $\rightarrow$ *wrong tagging decision*
    * $\pi^- \rightarrow d = +1, \text{label} = +1$ $\rightarrow$ *correct tagging decision*
* Assuming $B^-, SS\pi$
    * $\pi^+ \rightarrow d = -1, \text{label} = +1$ $\rightarrow$ *correct tagging decision*
    * $\pi^- \rightarrow d = +1, \text{label} = -1$ $\rightarrow$ *wrong tagging decision*
* Assuming $B^0, SS\pi$
    * $\pi^+ \rightarrow d = +1, \text{label} = +1$ $\rightarrow$ *correct tagging decision*
    * $\pi^- \rightarrow d = -1, \text{label} = -1$ $\rightarrow$ *wrong tagging decision*
* Assuming $\bar{B^0}, SS\pi$
    * $\pi^+ \rightarrow d = +1, \text{label} = -1$ $\rightarrow$ *wrong tagging decision*  
    * $\pi^- \rightarrow d = -1, \text{label} = +1$ $\rightarrow$ *correct tagging decision*

*It will be rescaled to 0
## Mistag
**$\eta$  = mistag ** 
It's the probability of assigning a wrong tag. 
If $\eta$ > 0.5, the tagging decision $d$ is flipped and the new mistag is $\eta'=1-\eta$. 
$\eta$ is obtained from the output of the NN that gives the probability of getting label = 1.
``` 
 NNoutput = prob[label=1] = prob[d=Q(B)]
mistag = 1 - NNout = prob[label=0] = prob[d*(-1)=Q(B)] 
```
