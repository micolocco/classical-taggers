import ROOT
import os
from ROOT import RDataFrame
import sys
import math
#from BranchesAndSelection import *
from array import array
from math import log, fabs, sqrt, cos, atan, exp, tan, acos, sin
#import TMVAClassification_MVA_class_code
from tqdm import tqdm
from ROOT import TFile, TFileMerger, TTree, TBranch, TMath
from builtins import max, min
import pickle
import numpy as np
import xgboost as xgb
import threading
import time
import psutil
from contextlib import contextmanager
import concurrent.futures
import traceback
import re


K_MASS  = 493.7
P_MASS  = 938.272
Pi_MASS  = 139.570
Ds_MASS = 1968.35
D_MASS = 1869.66
Lc_MASS = 2286.46
K_MASS_SQ  = K_MASS  * K_MASS
P_MASS_SQ  = P_MASS  * P_MASS
Pi_MASS_SQ  = Pi_MASS  * Pi_MASS
Ds_MASS_SQ  = Ds_MASS * Ds_MASS
D_MASS_SQ  = D_MASS * D_MASS
Lc_MASS_SQ  = Lc_MASS * Lc_MASS

label_map = {
        'lab0': 'B',
        'lab1': 'piplus',
        'lab2': 'Ds',
        'lab3': 'hplus',
        'lab4': 'hminus',
        'lab5': 'piminus',
    }
# Following definition at https://gitlab.cern.ch/lhcb-b2oc/analyses/b2dx-early-measurements/-/blob/master/BranchesAndSelection.py?ref_type=heads

BDT_branches = [
        "lab0_CHI2DOF", "lab0_IPCHI2_OWNPV", "eventNumber",
        "lab0_DIRA_OWNPV_Transformed", "lab0_MINIPCHI2_Log", "lab0_RFD",
        "lab0_VCHI2NDOF_Log", "lab0_LifetimeFit_VCHI2NDOF_Log",
        "lab2_DIRA_ORIVX_Transformed", "lab2_MINIPCHI2_Log", "lab2_RFD",
        "lab2_VCHI2NDOF_Log", "lab1_MINIPCHI2_Log", "lab1_PT",
        "lab1_CosTheta", "lab345_MIN_PT", "lab345_MIN_MINIPCHI2_Log",
        "lab1345_TRACK_GhostProb"
    ]
def extract_variables(expr):
    """Extract variable names from a transformation expression."""
    # This regex matches variable names composed of letters, digits, and underscores,
    # but excludes Python keywords like 'log', 'sqrt', 'pow', etc.
    reserved_words = {'log', 'sqrt', 'pow', 'min', 'max', 'abs', 'GetTheS'}
    variables = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
    return variables - reserved_words

def get_required_branches(needed_branch_expressions):
    """Get all the real branches required to compute the given derived variables."""
    required = set()
    for expr in needed_branch_expressions.values():
        required |= extract_variables(expr)
    return required

def optimize_tree_reading(inTree, needed_branch_expressions):
    """Enable only the branches required to compute the expressions."""
    inTree.SetBranchStatus("*", 0)

    required_branches = get_required_branches(needed_branch_expressions)

    for branch in required_branches:
        inTree.SetBranchStatus(branch, 1)
    
    return inTree

def compute_derived_variables(tree, needed_branches):
    """Compute derived variables from a ROOT TTree."""
    import ROOT

    n_entries = tree.GetEntries()
    result = {key: [] for key in needed_branches}

    for i in range(n_entries):
        tree.GetEntry(i)
        
        # Build context: var_name -> value
        context = {}
        for branch in get_required_branches(needed_branches):
            context[branch] = getattr(tree, branch)

        # Evaluate each derived variable
        for new_var, expr in needed_branches.items():
            try:
                value = evaluate_expression(expr, context)
            except Exception as e:
                print(f"Error evaluating {new_var} at entry {i}: {e}")
                value = float('nan')
            result[new_var].append(value)
    
    return result  # Or return pandas.DataFrame(result) if preferred


@ROOT.Numba.Declare(['float'], 'float')
def sq(sth):
    return sth**2


@contextmanager
def timing(description):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description}: {elapsed:.2f} seconds")

def memory_usage():
    """Get current memory usage."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # in MB

def monitor_worker_performance(start, end, inTree, tree_list, idx, pbar, pbar_lock, 
                              update_interval=1000, batch_size=10000):
    """Wrapper with performance monitoring."""
    print(f"Worker {idx} starting - Memory: {memory_usage():.2f} MB")
    
    with timing(f"Worker {idx} processing {end-start} events"):
        # Call the actual worker
        worker(start, end, inTree, tree_list, idx, pbar, pbar_lock, 
               update_interval, batch_size)
    
    print(f"Worker {idx} finished - Memory: {memory_usage():.2f} MB")
    # Analyze worker's tree
    if tree_list[idx]:
        print(f"Worker {idx} processed {tree_list[idx].GetEntries()} entries")

# To use this in BDTLoop, replace the worker call with monitor_worker_performance

cpp_code = """
// Function definition
template <typename F = Double_t>
F combined_mass(const std::vector<F>& masses, const std::vector<F>& PXs,
                const std::vector<F>& PYs, const std::vector<F>& PZs) {
  assert(masses.size() == PXs.size() && PXs.size() == PYs.size() &&
         PYs.size() == PZs.size());
  F e_sum = 0, px_sum = 0, py_sum = 0, pz_sum = 0;

  for (size_t i = 0; i < masses.size(); ++i) {
    auto m = masses[i], px = PXs[i], py = PYs[i], pz = PZs[i];
    e_sum += sqrt(m * m + px * px + py * py + pz * pz);
    px_sum += px;
    py_sum += py;
    pz_sum += pz;
  }
  return sqrt(e_sum * e_sum - px_sum * px_sum - py_sum * py_sum -
              pz_sum * pz_sum);
};
"""
ROOT.gInterpreter.ProcessLine(cpp_code)


cpp_code2 = """
Double_t K_MASS  = 493.7;
Double_t P_MASS  = 938.272;
Double_t Pi_MASS  = 139.570;
Double_t PI_MASS = Pi_MASS;
Double_t Ds_MASS = 1968.35;
Double_t D_MASS = 1869.66;
Double_t Lc_MASS = 2286.46;
Double_t K_MASS_SQ  = K_MASS  * K_MASS;
Double_t P_MASS_SQ  = P_MASS  * P_MASS;
Double_t PI_MASS_SQ  = Pi_MASS  * Pi_MASS;
Double_t Pi_MASS_SQ = PI_MASS_SQ;
Double_t Ds_MASS_SQ  = Ds_MASS * Ds_MASS;
Double_t D_MASS_SQ  = D_MASS * D_MASS;
Double_t Lc_MASS_SQ  = Lc_MASS * Lc_MASS;
"""
ROOT.gInterpreter.ProcessLine(cpp_code2)


cpp_code3 = """
double transformDIRA(const double& DIRAVal)
{
  assert(DIRAVal > -1 && DIRAVal < 1);
  if(DIRAVal > 0)
    return -std::log(1 - DIRAVal);
  else
    return std::log(1 + DIRAVal);
}
"""
ROOT.gInterpreter.ProcessLine(cpp_code3)


cpp_code_sq = """

Float_t sq(Float_t x){return x*x;}
"""

ROOT.gInterpreter.ProcessLine(cpp_code_sq)


cpp_CosTheta = """
// Function definition
Float_t GetTheS(Float_t MR, Float_t PXR, Float_t PYR, Float_t PZR, Float_t MP, Float_t PXP, Float_t PYP, Float_t PZP){


  Float_t ER = sqrt(PXR*PXR+PYR*PYR+PZR*PZR+MR*MR);
  Float_t EP = sqrt(PXP*PXP+PYP*PYP+PZP*PZP+MP*MP);

  TLorentzVector pR(PXR,PYR,PZR,ER);
  TLorentzVector pP(PXP,PYP,PZP,EP);

  // pR.Boost( -pR.BoostVector() ); cout << "GetTheS: " << pR.X() << " " << pR.Y() << " " << pR.Z() << endl;
  pP.Boost( -pR.BoostVector() );
  // Float_t thetas=(pP.Pz()/pP.P()); // cout << "GetTheS: " << pP.CosTheta() << " " << thetas << endl;
  return float(pP.CosTheta());

}
"""
ROOT.gInterpreter.ProcessLine(cpp_CosTheta)


# // from 2022RunsValidation_3_5_2023-selected.txt
cpp_GoodConditions = """
int GoodConditions_function(const int& runnumber_value){


    list<int> run_number_list = {
    256290,
256289,
256288,
256287,
256283,
256278,
256276,
256274,
256273,
256272,
256267,
256264,
256263,
256261,
256259,
256258,
256254,
256214,
256169,
256166,
256163,
256159,
256156,
256150,
256146,
256143,
256142,
256135,
256134,
256132,
256129,
256119,
256112,
256111,
256110,
256107,
256105,
256104,
256032,
256030,
256028,
256026,
256022,
256020,
256015,
256011,
256007,
256004,
256003,
256001,
256000,
255999,
255984,
255982,
255980,
255977,
255976,
255974,
255972,
255971,
255970,
255964,
255958,
255956,
255950,
255949
};


    bool found = false;

    for (auto it = run_number_list.begin(); it != run_number_list.end(); ++it) {
        if (*it == runnumber_value) {
            found = true;
            break;
        }
    }
    return found;
}
"""
ROOT.gInterpreter.ProcessLine(cpp_GoodConditions)


# cpp_opening_function = """
# #import <vector>
# vector<int> opening_function(){
#    // Open the input file
#     ifstream inputFile("2022RunsListGoodCondition_dedicatedVELOalignment_updated.txt");
#
#     // Create a vector to store the numbers
#     vector<int> data;
#
#     // Read each line from the file
#     string line;
#     while (getline(inputFile, line)) {
#         // Create a stringstream to extract numbers from the line
#         stringstream ss(line);
#
#         // Read each comma-separated number from the line
#         string numStr;
#         while (getline(ss, numStr, ',')) {
#             // Convert the string to an int and add it to the data vector
#             int num = stod(numStr);
#             data.push_back(num);
#         }
#     }
#
#     // Print the data vector
#     for (auto num : data) {
#         cout << num << " ";
#     }
#     cout << endl;
#
#     return data;
# }
# """
# ROOT.gInterpreter.ProcessLine(cpp_opening_function)
#
#
#
# cpp_GoodConditions = """
# vector<int> run_number_vector = opening_function();
#
# int GoodConditions_function(const int& runnumber_value){
#
#     bool found = false;
#
#     for (auto it = run_number_vector.begin(); it != run_number_vector.end(); ++it) {
#         if (*it == runnumber_value) {
#             found = true;
#             break;
#         }
#     }
#     return found;
# }
# """
# ROOT.gInterpreter.ProcessLine(cpp_GoodConditions)


# Load the classifiers
with open('Bs2DsPi_selections/MVA/classifier-split0.pkl', 'rb') as f:
    clf0 = pickle.load(f)

with open('Bs2DsPi_selections/MVA/classifier-split1.pkl', 'rb') as f:
    clf1 = pickle.load(f)


def Decorator(text = None):
    if text == None:
        print(
        """
----------------------------------------------------------------
----------------------------------------------------------------
        """)
    else:
        print(
        "----------------------------------------------------------------\n" \
        "{}\n" \
        "----------------------------------------------------------------".format(text)
    )


def Check(fileNameIn, treeIn, fileNameOut, treeOut):
    print("Checking the number of entries!")
    print("[INFO] Input file name: {}".format(fileNameIn))
    ROOT.ROOT.EnableImplicitMT()
    nb_in = RDataFrame(treeIn, fileNameIn).Count().GetValue()
    nb_out = RDataFrame(treeOut, fileNameOut).Count().GetValue()

    if treeIn != None:
        print("[INFO] Read {} TTree from: {}".format(treeIn, fileNameIn))
    else:
        raise ValueError("[ERROR] Couldn't read {} TTree from: {}".format(treeIn, fileNameIn))

    if treeOut != None:
        print("[INFO] Read {} TTree from: {}".format(treeOut, fileNameOut))
    else:
        raise ValueError("[ERROR] Couldn't read {} TTree from: {}".format(treeOut, fileNameOut))

    print("[INFO] Checking NTuples...")
    if nb_in == nb_out:
        print("[INFO] Everything is fine: {} == {}".format(nb_in, nb_out))
    else:
        raise ValueError("[ERROR] Number of entries different for {} and {}! Please check your MINI production! {} != {}".format(treeIn, treeOut, nb_in, nb_out))




def worker(chunk_id, start, end, inTree, tree_dict, pbar, pbar_lock):
    """
    Process a smaller chunk of events and update progress frequently.
    
    Args:
        chunk_id: Unique identifier for this chunk
        start, end: Range of entries to process
        inTree: Input ROOT tree
        tree_dict: Dictionary to store output trees (thread-safe)
        pbar, pbar_lock: Progress bar and its lock
    """
    # Create a unique local tree for this chunk
    localTree = inTree.CloneTree(0)
    local_pred = array('f', [-2.0])
    localTree.Branch("BDTGResponse_XGB_1", local_pred, 'BDTGResponse_XGB_1/F')
    
    chunk_size = end - start
    processed = 0
    
    try:
        # Process entries in mini-batches to update progress more frequently
        MINI_BATCH = 100  # Update progress every 100 entries
        
        inputs_batch = []
        selectors_batch = []
        
        for jentry in range(start, end):
            # Load the entry
            if inTree.LoadTree(jentry) < 0:
                break
                
            inTree.GetEntry(jentry)
            processed += 1
            
            # Apply cuts
            if (inTree.lab0_CHI2DOF > 20.0 or inTree.lab0_CHI2DOF < 0.0 or
                inTree.lab0_IPCHI2_OWNPV < 0.0):
                # Update progress bar every MINI_BATCH entries
                if processed % MINI_BATCH == 0:
                    with pbar_lock:
                        pbar.update(MINI_BATCH)
                continue
                
            # Collect features
            event_data = [
                inTree.lab0_DIRA_OWNPV_Transformed,
                inTree.lab0_MINIPCHI2_Log,
                inTree.lab0_RFD,
                inTree.lab0_VCHI2NDOF_Log,
                inTree.lab0_LifetimeFit_VCHI2NDOF_Log,
                inTree.lab2_DIRA_ORIVX_Transformed,
                inTree.lab2_MINIPCHI2_Log,
                inTree.lab2_RFD,
                inTree.lab2_VCHI2NDOF_Log,
                inTree.lab1_MINIPCHI2_Log,
                inTree.lab1_PT,
                inTree.lab1_CosTheta,
                inTree.lab345_MIN_PT,
                inTree.lab345_MIN_MINIPCHI2_Log,
                inTree.lab1345_TRACK_GhostProb
            ]
            
            inputs_batch.append(event_data)
            selectors_batch.append(inTree.eventNumber % 2)
            
            # Process mini-batch when we have enough entries
            if len(inputs_batch) >= MINI_BATCH:
                _process_mini_batch(inputs_batch, selectors_batch, localTree, local_pred)
                inputs_batch = []
                selectors_batch = []
            
            # Update progress bar every MINI_BATCH entries
            if processed % MINI_BATCH == 0:
                with pbar_lock:
                    pbar.update(MINI_BATCH)
        
        # Process remaining entries in the batch
        if inputs_batch:
            _process_mini_batch(inputs_batch, selectors_batch, localTree, local_pred)
            
        # Update remaining progress
        remaining = processed % MINI_BATCH
        if remaining > 0:
            with pbar_lock:
                pbar.update(remaining)
        
        # Store the result in the dictionary with the chunk ID as key
        with pbar_lock:  # Use the lock to modify the shared dictionary
            tree_dict[chunk_id] = localTree
            
    except Exception as e:
        print(f"Error in chunk {chunk_id} (entries {start}-{end}): {str(e)}")
        traceback.print_exc()
        
        # Update progress bar for remaining entries
        with pbar_lock:
            remaining = end - start - processed
            if remaining > 0:
                pbar.update(remaining)
    
    return chunk_id, processed

def _process_mini_batch(inputs, selectors, tree, pred_array):
    """Process a mini-batch of events efficiently"""
    if not inputs:
        return
        
    # Convert to numpy arrays
    X = np.array(inputs)
    selectors = np.array(selectors)
    
    # Create boolean masks for even/odd selectors
    mask_even = (selectors == 0)
    mask_odd = ~mask_even
    
    # Predict with appropriate model
    preds = np.zeros(len(inputs), dtype=np.float32)
    
    if np.any(mask_even):
        preds[mask_even] = clf1.predict_proba(X[mask_even])[:, 1]
    if np.any(mask_odd):
        preds[mask_odd] = clf0.predict_proba(X[mask_odd])[:, 1]
    
    # Fill tree with predictions
    for p in preds:
        pred_array[0] = p
        tree.Fill()

def calculate_responses(start, end, inTree, responses, indices, pbar, pbar_lock):
    """
    Calculate BDT responses for a range of entries and store them in shared arrays.
    
    Args:
        start, end: Range of entries to process
        inTree: Input ROOT tree
        responses: Array to store calculated responses
        indices: Array to store indices of valid events
        pbar, pbar_lock: Progress bar and its lock
    """
    processed = 0
    valid_count = 0
    
    try:
        # Use a smaller batch size for more frequent progress updates
        MINI_BATCH = 100
        batch_inputs = []
        batch_selectors = []
        batch_indices = []
        
        for jentry in range(start, end):
            # Load the entry
            if inTree.LoadTree(jentry) < 0:
                break
                
            inTree.GetEntry(jentry)
            processed += 1
            
            # Apply cuts
            if (inTree.lab0_CHI2DOF > 20.0 or inTree.lab0_CHI2DOF < 0.0 or
                inTree.lab0_IPCHI2_OWNPV < 0.0):
                # Update progress bar every MINI_BATCH entries
                if processed % MINI_BATCH == 0:
                    with pbar_lock:
                        pbar.update(MINI_BATCH)
                continue
                
            # Collect features
            event_data = [
                inTree.lab0_DIRA_OWNPV_Transformed,
                inTree.lab0_MINIPCHI2_Log,
                inTree.lab0_RFD,
                inTree.lab0_VCHI2NDOF_Log,
                inTree.lab0_LifetimeFit_VCHI2NDOF_Log,
                inTree.lab2_DIRA_ORIVX_Transformed,
                inTree.lab2_MINIPCHI2_Log,
                inTree.lab2_RFD,
                inTree.lab2_VCHI2NDOF_Log,
                inTree.lab1_MINIPCHI2_Log,
                inTree.lab1_PT,
                inTree.lab1_CosTheta,
                inTree.lab345_MIN_PT,
                inTree.lab345_MIN_MINIPCHI2_Log,
                inTree.lab1345_TRACK_GhostProb
            ]
            
            batch_inputs.append(event_data)
            batch_selectors.append(inTree.eventNumber % 2)
            batch_indices.append(jentry)
            
            # Process mini-batch when we have enough entries
            if len(batch_inputs) >= MINI_BATCH:
                process_batch(batch_inputs, batch_selectors, batch_indices, responses)
                valid_count += len(batch_inputs)
                
                # Store indices for valid events
                for idx in batch_indices:
                    indices.append(idx)
                
                batch_inputs = []
                batch_selectors = []
                batch_indices = []
                
            # Update progress bar every MINI_BATCH entries
            if processed % MINI_BATCH == 0:
                with pbar_lock:
                    pbar.update(MINI_BATCH)
        
        # Process remaining entries in the batch
        if batch_inputs:
            process_batch(batch_inputs, batch_selectors, batch_indices, responses)
            valid_count += len(batch_inputs)
            
            # Store indices for valid events
            for idx in batch_indices:
                indices.append(idx)
            
        # Update remaining progress
        remaining = processed % MINI_BATCH
        if remaining > 0:
            with pbar_lock:
                pbar.update(remaining)
                
    except Exception as e:
        print(f"Error in worker processing entries {start}-{end}: {str(e)}")
        traceback.print_exc()
        
        # Update progress bar for remaining entries
        with pbar_lock:
            remaining = end - start - processed
            if remaining > 0:
                pbar.update(remaining)
    
    return valid_count

def process_batch(inputs, selectors, indices, responses_dict):
    """Process a batch of events and store results in the responses dictionary"""
    if not inputs:
        return
        
    # Convert to numpy arrays
    X = np.array(inputs)
    selectors = np.array(selectors)
    
    # Create boolean masks for even/odd selectors
    mask_even = (selectors == 0)
    mask_odd = ~mask_even
    
    # Predict with appropriate model
    preds = np.zeros(len(inputs), dtype=np.float32)
    
    if np.any(mask_even):
        preds[mask_even] = clf1.predict_proba(X[mask_even])[:, 1]
    if np.any(mask_odd):
    
    def BDTLoop(infile, inTree, outFile, outTree, nMax):
    """
    Simplified BDTLoop that calculates responses for all events first,
    then creates a copy of the input tree with the additional response branch.
    """
    print(f"Starting the BDT processing for {infile}")
    
    # Initialize global classifier models if not already done
    if 'clf0' not in globals() or 'clf1' not in globals():
        init_models()
    
    # Optimize tree reading
    inTree_shortened = inTree.CloneTree()
    optimize_tree_reading(inTree_shortened)
    
    # Determine number of entries to process
    nentries = inTree_shortened.GetEntries()
    if nMax < 0 or nMax > nentries:
        nMax = nentries
    
    # Shared dictionary to store BDT responses (thread-safe)
    # Using a Manager dictionary from multiprocessing would be more correct,
    # but for simplicity we'll use a regular dict with a lock
    responses = {}
    response_lock = threading.Lock()
    
    # Keep track of valid event indices
    valid_indices = []
    
    # Create progress bar
    pbar = tqdm(total=nMax, desc="Calculating BDT responses", mininterval=0.2)
    pbar_lock = threading.Lock()
    
    # Determine chunk size and number of threads
    CHUNK_SIZE = 10000  # Smaller chunks for more progress updates
    num_chunks = (nMax + CHUNK_SIZE - 1) // CHUNK_SIZE
    n_threads = min(max(1, os.cpu_count() - 1), 16)
    
    print(f"Processing {nMax} entries in {num_chunks} chunks using {n_threads} threads")
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            
            # Submit chunks for processing
            for i in range(num_chunks):
                start_idx = i * CHUNK_SIZE
                end_idx = min(start_idx + CHUNK_SIZE, nMax)
                
                # Create thread-local lists for each chunk
                chunk_responses = {}
                chunk_indices = []
                
                future = executor.submit(
                    calculate_responses, 
                    start_idx, end_idx, 
                    inTree_shortened, chunk_responses, chunk_indices,
                    pbar, pbar_lock
                )
                futures.append((future, chunk_responses, chunk_indices))
                
                # Small delay to prevent CPU spikes
                if i % n_threads == 0:
                    time.sleep(0.01)
            
            # Wait for all futures to complete
            for future, chunk_responses, chunk_indices in futures:
                try:
                    valid_count = future.result()
                    
                    # Merge responses and indices with global collections
                    with response_lock:
                        responses.update(chunk_responses)
                        valid_indices.extend(chunk_indices)
                        
                except Exception as e:
                    print(f"Error processing chunk: {e}")
    
    except Exception as e:
        print(f"Error in main thread: {str(e)}")
        traceback.print_exc()
        
    finally:
        pbar.close()
    
    print(f"Calculated BDT responses for {len(responses)} events")
    
    # Now create a new tree with the BDT response branch
    print("Creating output tree with BDT responses...")
    
    # Clone the input tree structure (this doesn't copy the entries yet)
    outTree = inTree.CloneTree(0)
    
    # Create branch for BDT response
    bdt_response = array('f', [-2.0])
    bdt_branch = outTree.Branch("BDTGResponse_XGB_1", bdt_response, 'BDTGResponse_XGB_1/F')
    
    # Process entries and fill the output tree
    copy_pbar = tqdm(total=nentries, desc="Copying tree with responses", mininterval=0.2)
    
    for i in range(nentries):
        inTree.GetEntry(i)
        
        # Set default response value
        bdt_response[0] = -2.0
        
        # If we have a response for this entry, use it
        if i in responses:
            bdt_response[0] = responses[i]
            
        # Fill the tree (this copies all branches from the current entry in inTree)
        outTree.Fill()
        
        if i % 10000 == 0:
            copy_pbar.update(min(10000, i) if i > 0 else 1)
    
    # Update remaining progress
    remaining = nentries % 10000
    if remaining > 0:
        copy_pbar.update(remaining)
    
    copy_pbar.close()
    
    # Write tree to output file
    print(f"Saving tree with {outTree.GetEntries()} entries")
    outTree.Write()
    outFile.Close()
    print("Processing completed successfully")

def optimize_tree_reading(inTree):
    """Optimize tree reading by enabling only necessary branches."""
    # Disable all branches first
    inTree.SetBranchStatus("*", 0)
    
    # Enable only the branches we need for BDT calculation
    needed_branches = [
        "lab0_CHI2DOF", "lab0_IPCHI2_OWNPV", "eventNumber",
        "lab0_DIRA_OWNPV_Transformed", "lab0_MINIPCHI2_Log", "lab0_RFD",
        "lab0_VCHI2NDOF_Log", "lab0_LifetimeFit_VCHI2NDOF_Log",
        "lab2_DIRA_ORIVX_Transformed", "lab2_MINIPCHI2_Log", "lab2_RFD",
        "lab2_VCHI2NDOF_Log", "lab1_MINIPCHI2_Log", "lab1_PT",
        "lab1_CosTheta", "lab345_MIN_PT", "lab345_MIN_MINIPCHI2_Log",
        "lab1345_TRACK_GhostProb"
    ]
    get_required_branches(needed_branch_expressions)

    for branch in needed_branches:
        inTree.SetBranchStatus(branch, 1)
    
    # Need to have all branches activated for the final copy
    # We'll re-enable all branches right before the tree copy phase
    
    return inTree

def init_models():
    """Initialize XGBoost models."""
    global clf0, clf1
    
    # This is a placeholder. In your actual code, 
    # you would load your models here.
    print("Initializing classification models...")
    
    # Example (replace with your actual model loading code):
    # clf0 = xgb.Booster()
    # clf0.load_model('model0.json')
    # clf1 = xgb.Booster()
    # clf1.load_model('model1.json')

def BDTCheck(fileNameIn, fileNameBDT):
    print("[INFO] Input file name: {}".format(fileNameIn))
    fileIn = TFile.Open(fileNameIn)
    treeIn = fileIn.Get("Tuple/DecayTree")

    if (treeIn != None):
        print("[INFO] Read TTree from: {}".format(fileNameIn))
    else:
        print("[ERROR] Couldn't read TTree from: {}".format(fileNameIn))

    print("[INFO] BDT Ntuple to check: {}".format(fileNameBDT))
    fileBDT = TFile.Open(fileNameBDT)
    treeBDT = fileBDT.Get("Tuple/DecayTree")
    if ( treeBDT != None):
        print("[INFO] Read TTree from: {}".format(fileNameBDT))
    else:
        print("[ERROR] Couldn't read TTree from: {}".format(fileNameBDT))

    print("[INFO] Checking NTuples ")
    treshold = treeIn.GetEntries()*0.0001

    if ( fabs(treeIn.GetEntries() - treeBDT.GetEntries()) < treshold ):
        print("[INFO] Everything is fine: {} == {}".format(treeIn.GetEntries(), treeBDT.GetEntries()))
        print("[INFO] Difference smaller than 0.01%: {} < {}".format(fabs(treeIn.GetEntries() - treeBDT.GetEntries()), treshold))
    else:
        print("[ERROR] Number of entries different! Please check your MINI production! {} != {}".format(treeIn.GetEntries(), treeBDT.GetEntries()))
        print("[ERROR] Difference larger than 0.01%: {} > {}".format(fabs(treeIn.GetEntries() - treeBDT.GetEntries()), treshold))


