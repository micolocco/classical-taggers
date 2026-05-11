````markdown
# Run 3 Flavour Tagging Pipeline

This repository contains the Snakemake workflows used to:

1. **Train** the Run 3 flavour–tagging algorithms on Monte Carlo and write the tagging decision + mistag back into ROOT ntuples.
2. **Apply** the trained taggers to **real data** for calibration, using the same preselection cuts and models derived from MC.

There are therefore two main Snakefiles:

- **MC training workflow**: `Snakefile` in the repository root  
- **Data calibration workflow**: `classical_taggers/data_calibration/Snakefile`

Both workflows are fully reproducible and can be executed locally or on a batch system via Snakemake.

---

## 1. MC Training Workflow (root `Snakefile`)

### 1.1 Pipeline at a glance

The MC workflow is structured in four main stages:

1. **Feature construction**  
   `1_raw/ → 2_added_features/`  
   Rule: `add_features`  
   Adds all derived variables needed for the decision tree (DT) and neural network (NN) training.

2. **Preselection optimisation (Decision Tree)**  
   `2_added_features/ → DT_outputs/`  
   Rule: `train_DT`  
   Trains a decision tree to derive rectangular preselection cuts per tagger.

3. **Application of preselection cuts**  
   `2_added_features/ → 3_selected/`  
   Rule: `add_selection`  
   Applies the DT cuts to create the training and hold-out samples.

4. **Neural network training and tagging**  
   - Training: `3_selected/ → savedModels/` (`train_tagger`)
   - Tagging: `3_selected/ + savedModels/ → 4_tagged/` (`add_tagDec`)

Final **tagged MC ntuples** live under `4_tagged/` and contain new branches with the tagging decision and mistag probability.

---

### 1.2 Decays and taggers (MC)

The mapping between decay channels and taggers is defined in the root `Snakefile` via:

```python
taggers_conf = {
    'Bu2JpsiK'   : ['OSKaon', 'OSElectron', 'OSMuon', 'SSPion', 'SSKaon', 'SSProton'],
    'Bd2JpsiKst' : ['SSPion', 'SSProton', 'OSKaon', 'OSElectron', 'OSMuon', 'SSKaon'],
    'Bs2DsPi'    : ['SSKaon', 'OSKaon', 'OSElectron', 'OSMuon'],
    'Bd2DmPi'    : ['SSPion', 'SSProton', 'OSKaon', 'OSElectron', 'OSMuon'],
    'Bs2JpsiPhi' : ['OSKaon', 'OSElectron', 'OSMuon', 'SSPion', 'SSProton', 'SSKaon']
}
````

This dictionary is used everywhere to build file paths and to loop over all relevant `{decay, tagger}` combinations.

For some steps, taggers are trained on a **different** decay than the one they are later evaluated on.
This is handled by:

```python
def extract_decay(tagger):
    if tagger in ['OSKaon', 'OSMuon', 'OSElectron']:
        return 'Bu2JpsiK'
    elif tagger == 'SSKaon':
        return 'Bs2DsPi'
    elif tagger in ['SSPion', 'SSProton']:
        return 'Bd2JpsiKst'
    else:
        raise ValueError(f"Unknown tagger: {tagger}")
```

---

### 1.3 Directory structure (MC)

The modified MC files follow a simple, stage-based directory layout under `${MODIFIED_MC}`:

```text
withUT_MC_2024/
  1_raw/             # copies of the central MC files
  2_added_features/  # ntuples with extra training features
  3_selected/        # events passing DT preselection cuts
  4_tagged/          # ntuples with tagging decision + mistag

DT_outputs/          # DT-based preselection cuts
savedModels/         # trained NN models and logs
paths_for_snakemake/ # helper text files listing input paths
configs/             # YAML configs for NN architectures/hyperparams
scripts/             # all Python scripts called by the rules
```

Different MC configurations (e.g. `withUT_MC_2024`, `noUT_MC_2024`) appear as the `sample_type` wildcard.

---

### 1.4 Configuration

Snakemake expects a configuration file (e.g. `config.yaml`) with at least:

```yaml
RAW_MC: "/path/to/raw/mc"          # base path for raw MC
MODIFIED_MC: "/path/to/modified/mc"
MODIFIED_DATA: "/path/to/modified/data"  # also used by data-calibration Snakefile
REPO: "/path/to/this/repo"
```

These values are loaded at the top of the Snakefiles, for example:

```python
raw_MC      = config['RAW_MC']
modified_MC = config['MODIFIED_MC']
modified_data = config['MODIFIED_DATA']  # used by data calibration
repo        = config['REPO']
```

Make sure these paths are set correctly before running the workflows.

---

### 1.5 Rules overview (MC)

#### `add_features` (MC)

* **Input**:
  `${RAW_MC}/{sample_type}/1_raw/{decay}/{id}.root`
  Script: `scripts/adding_features_v2.py`
* **Output**:
  `${MODIFIED_MC}/{sample_type}/2_added_features/{decay}/{id}.root`
* **What it does**:
  Reads the raw ntuple, computes all derived features used by the DT and NNs (PID variables, IP sigmas, kinematic variables, multiplicities, etc.), and writes a new ROOT file.
  The correct `DecayTree` name is chosen via a helper `find_tree_name(decay)`.

---

#### `train_DT`

* **Input**:
  Directory `${MODIFIED_MC}/{sample_type}/2_added_features/`
  Script: `scripts/origin_DT_cut.py`
* **Output**:
  Preselection cut files:

  ```text
  {MODIFIED_MC}/{sample_type}/DT_outputs/{cut_name}/{balanced}/cuts/{tagger}_preselections.txt
  ```
* **What it does**:
  Trains a decision tree to separate signal-like from background-like tagging particles and extracts rectangular preselection cuts per tagger.
  Cuts are stored as human-readable text files and later used by both MC training and data calibration pipelines.

---

#### `add_selection` (MC)

* **Input**:

  * `${MODIFIED_MC}/{sample_type}/2_added_features/{decay}/{id}.root`
  * Cut file (in the MC path):

    ```text
    {MODIFIED_MC}/withUT_MC_2024/DT_outputs/{cut_name}/balanced/cuts/{tagger}_preselections.txt
    ```
* **Output**:

  ```text
  {MODIFIED_MC}/{sample_type}/3_selected/{decay}/{tagger}/
      {cut_name}/{balanced}/{features}/{id}.root
  ```
* **What it does**:
  Applies the DT preselection using `scripts/preSelections.py`.
  Creates tagger-specific selected samples used for NN training and for the hold-out evaluation.
  One input file (e.g. ending in `02_1.mc.root`) is reserved as a **hold-out sample** and excluded from training.

---

#### `train_tagger`

* **Input**:
  All selected files for a given `{decay, tagger}` (except the hold-out), constructed from `ntuples_selected_withUT[...]`.
  Script: `scripts/pipeline.py`
* **Output** (example path):

  ```text
  {MODIFIED_MC}/savedModels/{sample_type}/{decay}/{tagger}/
      {cut_name}_{balanced}/{features}/{seed}/{config}/
      {asymmetry_level}/hold_out_bis/model.pth
  ```
* **What it does**:
  Trains the NN tagger for a given configuration (feature set, cuts, balanced/unbalanced, random seed, architecture, asymmetry level).
  Saves:

  * the trained model (`model.pth`),
  * training logs,
  * monitoring plots (loss curves, ROC curves, input-feature distributions).

---

#### `add_tagDec` (MC)

* **Input**:
  Hold-out selected file:

  ```text
  {MODIFIED_MC}/{sample_type}/3_selected/{decay}/{tagger}/
      {cut_name}/{balanced}/{features}/{id}.root
  ```

  Script: `scripts/adding_tagDec.py`
  Model path built using `modelPrePath` and `extract_decay(tagger)`.
* **Output**:

  ```text
  {MODIFIED_MC}/{sample_type}/4_tagged/{decay}/{tagger}/
      {cut_name}/{balanced}/{features}/{asym_level}/hold_out/{id}.root
  ```
* **What it does**:
  Loads the trained model, evaluates it on the hold-out sample, and writes new branches with tagging decision and mistag probability to the output ntuple.

---

### 1.6 Running the MC workflow

From the repository root:

```bash
snakemake -j 4
```

or explicitly:

```bash
snakemake -j 4 all
```

This will build the default targets defined in `rule all` (e.g. specific trained models for selected taggers).

#### Example: train a specific tagger configuration

```bash
snakemake \
  ${MODIFIED_MC}/savedModels/withUT_MC_2024/Bd2JpsiKst/SSPion/\
allBKGCAT_notSamePV_noOSP_SSK_balanced/union_PROBNN/45/\
lr0.001_bs4096_nL3_nN256/asym_level1/hold_out_bis/model.pth \
  -j 8
```

Snakemake will automatically run all prerequisite rules (`add_features`, `train_DT`, `add_selection`, `train_tagger`, `add_tagDec` if requested).

#### Running on a batch system (HTCondor example)

Adapt to your setup:

```bash
snakemake -j 100 \
  --cluster "condor_submit -append 'request_memory={resources.mem_mb}' \
                           -append 'MaxRuntime = {resources.MaxRunHours} * 3600'"
```

Each rule’s `resources` block already provides `mem_mb` and `MaxRunHours` hints.

---

### 1.7 Extending the MC pipeline

#### Add a new decay or tagger

1. Update `taggers_conf` in the root `Snakefile`.
2. Add raw MC paths for the new decay to `ntuples_eos_withUT`.
3. If needed, update `find_tree_name(decay)` with the correct TTree path.
4. Adjust any scripts in `scripts/` that have hard-coded decay lists or branch names.

#### Add a new feature set

1. Define a new feature group name (e.g. `union_PROBNN_v2`).
2. Make sure `scripts/adding_features_v2.py` and the training script understand that feature set.
3. Use the new `{features}` wildcard when calling `snakemake`.

---

## 2. Data Calibration Workflow (`classical_taggers/data_calibration/Snakefile`)

The second Snakefile applies the **MC-trained taggers** to **real data** for calibration.
It reuses:

* the same feature construction script,
* the same preselection cuts derived on MC,
* and the same trained NN models,

but works on data ntuples instead of MC.

---

### 2.1 Scope and taggers (data)

For data, the taggers currently configured are:

```python
taggers_conf = {
    'Bu2JpsiK'   : ['OSKaon', 'OSElectron', 'OSMuon'],
    'Bd2JpsiKst' : ['SSPion', 'SSProton', 'OSKaon', 'OSElectron', 'OSMuon'],
    'Bs2DsPi'    : ['SSKaon', 'OSKaon', 'OSElectron', 'OSMuon'],
    # 'Bd2DmPi': [...]
    # 'Bs2JpsiPhi': [...]
}
```

Again, the training decay associated with each tagger is obtained via the same `extract_decay(tagger)` function:

```python
def extract_decay(tagger):
    if tagger in ['OSKaon', 'OSMuon', 'OSElectron']:
        return 'Bu2JpsiK'
    elif tagger == 'SSKaon':
        return 'Bs2DsPi'
    elif tagger in ['SSPion', 'SSProton']:
        return 'Bd2JpsiKst'
    else:
        raise ValueError(f"Unknown tagger: {tagger}")
```

---

### 2.2 Data input lists and 1_raw (data)

Input data files are defined via text lists such as:

```text
block12_list_Bu2JpsiK.txt
block12_list_Bd2JpsiKst.txt
block12_list_Bs2DsPi.txt
```

Each file contains one EOS path per line, for example:

```text
/eos/lhcb/grid/prod/lhcb/anaprod/lhcb/LHCb/Collision24/DATA24.ROOT/...
```

The helper function:

```python
def select_input_files(decay):
    eosdir = 'root://eoslhcb.cern.ch/'
    txt = f"block12_list_{decay}.txt"
    with open(txt, "r") as f:
        lines = [line.strip() for line in f]
    return [eosdir + path for path in lines]
```

builds the full EOS URLs for each decay.

From that, the Snakefile constructs the **local raw data paths** under `${MODIFIED_DATA}`:

```python
ntuples_raw_withUT = {
    decay: [
        join(data, 'withUT_MC_2024', '1_raw', decay, basename(path))
        for path in select_input_files(decay)
    ]
    for decay in taggers_conf
}
```

So the data directory structure under `${MODIFIED_DATA}` mirrors the MC layout:

```text
MODIFIED_DATA/
  withUT_MC_2024/
    1_raw/{decay}/*.root          # stripped data copied locally
    2_added_features/{decay}/...
    3_selected/{decay}/{tagger}/...
    4_tagged/{decay}/{tagger}/...
```

---

### 2.3 Rules overview (data calibration)

All rules live in `classical_taggers/data_calibration/Snakefile`.

#### `rule all` (data)

The default target expands all 4_tagged files for the configured taggers, e.g. for `Bs2DsPi`:

```python
rule all:
    input:
        expand(ntuples_tagged_withUT['Bs2DsPi']['OSMuon'],
               cut_name=['allBKGCAT_notSamePV_noOSP_SSK'],
               balanced=['balanced'],
               features=['union_PROBNN'],
               asym_level=['asym_level1']),
        expand(ntuples_tagged_withUT['Bs2DsPi']['OSElectron'], ...),
        expand(ntuples_tagged_withUT['Bs2DsPi']['OSKaon'], ...),
        expand(ntuples_tagged_withUT['Bs2DsPi']['SSKaon'], ...)
```

Here, `ntuples_tagged_withUT` defines the expected 4_tagged paths:

```python
ntuples_tagged_withUT[decay][tagger]
    -> modified_data/withUT_MC_2024/4_tagged/{decay}/{tagger}/
       {cut_name}/{balanced}/{features}/{asym_level}/{id}.root
```

---

#### `add_features` (data)

```python
rule add_features:
    input:
        script = join(repo, 'scripts/adding_features_v2.py'),
        raw    = join(data, '{sample_type}/1_raw/{decay}/{id}.root')
    output:
        root   = join(modified_data,
                      '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/'
                      '2_added_features/{decay}/{id,.*}.root')
    log:
        join(modified_data,
             '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/'
             '2_added_features/{decay}/.{id,.*}.log')
    ...
    run:
        tree = find_tree_name(wildcards.decay)
        cmd = [
            'python', input.script,
            '--raw {input.raw}',
            '--output {output.root}',
            '--evtType {wildcards.decay}',
            '--treename', tree,
            '--data_calib',
            '&> {log}',
        ]
        shell(' '.join(cmd))
```

**Key points:**

* Uses the same `adding_features_v2.py` as the MC pipeline.
* Uses a simplified `find_tree_name` (for data only `Bs2DsPi` is special; other decays default to `Tuple/DecayTree`).
* Adds the flag `--data_calib` so the script can adjust behaviour for real data if needed.
* Writes to `${MODIFIED_DATA}/.../2_added_features`.

---

#### `add_selection` (data)

```python
rule add_selection:
    input:
        script         = join(repo, 'scripts/preSelections.py'),
        added_features = join(modified_data,
                              '{sample_type}/2_added_features/{decay}/{id}.root'),
        cut_file       = join(modified_MC,
                              'withUT_MC_2024/DT_outputs/{cut_name}/balanced/'
                              'cuts/{tagger}_preselections.txt')
    output:
        join(modified_data,
             '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/3_selected/'
             '{decay}/{tagger}/{cut_name}/{balanced}/{features}/{id,.*}.root')
    ...
    run:
        cmd = [
            'python', input.script,
            '--added_features {input.added_features}',
            '--output {output}',
            '--cut_file', input.cut_file,
            '--evtType {wildcards.decay}',
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--data_calib',
            '--repo {repo}',
            '&> {log}',
        ]
        shell(' '.join(cmd))
```

**Key points:**

* **Reuses the DT cuts from MC** via `${MODIFIED_MC}/withUT_MC_2024/DT_outputs/...`.
* Applies exactly the same preselection as used for training.
* Writes selected data ntuples to `${MODIFIED_DATA}/.../3_selected/...`.
* Uses `--data_calib` to tag this as data processing in the script.

---

#### `add_tagDec` (data)

```python
rule add_tagDec:
    input:
        script   = join(repo, 'scripts/adding_tagDec.py'),
        selected = join(modified_data,
                        '{sample_type}/3_selected/{decay}/{tagger}/'
                        '{cut_name}/{balanced}/{features}/{id}.root')
    output:
        root = join(modified_data,
            '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_tagged/'
            '{decay}/{tagger}/{cut_name}/{balanced}/{features}/'
            '{asym_level}/{id}.root'),
    params:
        modelPrePath = lambda wildcards: join(
            modified_MC,
            f'savedModels/{wildcards.sample_type}/'
            f'{extract_decay(wildcards.tagger)}/{wildcards.tagger}/'
            f'{wildcards.cut_name}_{wildcards.balanced}/'
            f'{wildcards.features}'
        ),
        taggedDataPath = join(modified_data,
            '{sample_type,(withUT_MC_2024|noUT_MC_2024)}/4_tagged/'
            '{decay}/{tagger}/{cut_name}/{balanced}/{features}/{asym_level}/'),
    ...
    run:
        cmd = [
            'python', input.script,
            '--selected {input.selected}',
            '--cut {wildcards.cut_name}_{wildcards.balanced}',
            '--taggedData {output.root}',
            '--modelPrePath {params.modelPrePath}',
            '--decayType {wildcards.decay}',
            '--tagger {wildcards.tagger}',
            '--features {wildcards.features}',
            '--asymmetry_level {wildcards.asym_level}',
            '--repo', repo,
            '--data_calib',
            '&> {log}'
        ]
        shell(' '.join(cmd))
```

**Key points:**

* Uses **trained models from MC**: `${MODIFIED_MC}/savedModels/...`
  The training decay used in the model path is determined by `extract_decay(tagger)`.
* Applies the NN tagger to data `3_selected` ntuples and writes to `${MODIFIED_DATA}/4_tagged/...`.
* Adds the `--data_calib` flag to distinguish from MC usage if needed in `adding_tagDec.py`.

Result: final **tagged data ntuples** ready for calibration and physics fits.

---

### 2.4 Running the data-calibration workflow

From the `classical_taggers/data_calibration` directory:

```bash
cd classical_taggers/data_calibration

# Dry-run to see the DAG
snakemake -n

# Actual run (local)
snakemake -j 4
```

or, for a larger parallel execution:

```bash
snakemake -j 50
```

To run on a batch system, you can use the same `--cluster` pattern as for the MC workflow, but from within this subdirectory:

```bash
snakemake -j 100 \
  --cluster "condor_submit -append 'request_memory={resources.mem_mb}' \
                           -append 'MaxRuntime = {resources.MaxRunHours} * 3600'"
```

Make sure that:

* `config.yaml` is accessible (either via `--configfile` or default),
* `MODIFIED_DATA`, `MODIFIED_MC` and `REPO` are set correctly,
* the files `block12_list_<decay>.txt` exist in `classical_taggers/data_calibration/`.

---

## 3. References

* [Snakemake documentation](https://snakemake.readthedocs.io/)
* Internal LHCb documentation / thesis sections describing:

  * flavour-tagging strategy,
  * training setup on MC,
  * calibration procedure on data using the tagged ntuples produced by this pipeline.

```
```
