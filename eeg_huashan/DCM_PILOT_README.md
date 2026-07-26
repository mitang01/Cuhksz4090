# Two-participant picture-naming DCM pilot

`run_dcm_pilot.py` runs the requested pilot without representing ordinary
connectivity as DCM:

1. MNE-Python reads and preprocesses the four BrainVision recordings.
2. Picture trials are classified from the original marker descriptions.
3. Picture-onset ERPs and within-recording sensor statistics are generated.
4. Clean epochs are exported to EEGLAB format.
5. MATLAB/SPM12 performs actual DCM-ERP and DCM-CSD inversion.
6. Model evidence, posterior probabilities, connection estimates, and 90%
   credible intervals are exported.

## Input data

The built-in manifest uses:

| Participant | Task | BrainVision header |
|---|---|---|
| Pp1_WWL | Picture naming | `20260630_picNaming_001_WWL.vhdr` |
| Pp1_WWL | Rest | `20260630_rest_001_WWL.Vhdr` |
| Pp2_JYP | Picture naming | `20260702_picNaming_001_JYP.vhdr` |
| Pp2_JYP | Rest | `20260702_rest_001_JYP.vhdr` |

All paths are the exact `/share/workspace2/tangmi/eeg_huashan/...` paths
provided for this pilot. Keep `.vhdr`, `.vmrk`, and `.eeg` files together.

## Marker rules

- `1`: picture onset and ERP time zero
- `3`: picture offset
- `2`: correct delayed verbal response
- `4`: premature verbal response

Every approved `Stimulus/... 1` starts a trial. Its first subsequent marker 2
or 4 closes it. A new marker 1 closes an unfinished preceding trial as
missing, then starts a new one. The script saves this classification to
`picture_trials.csv`; inspect it before interpreting results.

Two picture datasets are analyzed separately:

- `picture_all`: every marker-1 onset, including premature/missed trials
- `picture_correct`: marker-1 onsets whose first terminal response is marker 2

## Installation

Python requirements:

```bash
cd /share/workspace2/tangmi/eeg_huashan
python3 -m pip install -r requirements.txt
```

DCM additionally requires:

- MATLAB
- SPM12 on the remote machine

Download SPM from its official open-source distribution and record the exact
revision used. Set its location:

```bash
export SPM12_PATH=/path/to/spm12
```

## Run

```bash
cd /share/workspace2/tangmi/eeg_huashan
python3 run_dcm_pilot.py
```

To produce ERP/QC results and SPM-ready exports before MATLAB/SPM is installed:

```bash
python3 run_dcm_pilot.py --skip-dcm
```

Existing results are protected. Regenerate them explicitly:

```bash
python3 run_dcm_pilot.py --overwrite
```

## Outputs

Results are written under `dcm_pilot_results/<participant>/`.

Picture outputs include:

- `picture_trials.csv`
- `picture_all_erp.png`
- `picture_correct_erp.png`
- temporal-cluster and descriptive-statistics CSV files
- exact ROI-channel JSON files
- cleaned FIF and EEGLAB epochs

DCM outputs are separated into:

```text
dcm/picture_all/
dcm/picture_correct/
dcm/rest/
```

Each contains:

- all three fitted `DCM_*.mat` files
- model/family free energies and posterior probabilities
- a model-comparison plot
- winning-model connection means and 90% credible intervals
- a conventional network/parameter DCM plot

## DCM models

Four left-hemisphere template sources are used:

- occipitotemporal cortex (OT)
- posterior middle temporal gyrus (pMTG)
- anterior temporal lobe (ATL)
- inferior frontal gyrus (IFG)

Families:

- F1: `OT → pMTG/ATL → IFG`
- F2: F1 plus `IFG → pMTG/ATL` feedback
- F3: F1 plus direct `OT → IFG`

Picture epochs use DCM for evoked responses. Rest uses a **separate**
cross-spectral-density DCM. Rest is not treated as an ERP baseline because it
has no event-related input.

## Template choices and limits

The script tries MNE's `GSN-HydroCel-128` or `GSN-HydroCel-129` montage when
EGI E-numbered labels match. Otherwise, it matches labels to
`standard_1005`. It never assigns positions by channel order.

SPM's canonical MNI template head model is used with fixed ECD source priors.
Both are proxies. They cannot reproduce individual cap placement, skull/head
geometry, lesion anatomy, or postoperative anatomy. The command prints this
warning and records the exact template/matching result in audit and method
files.

Accordingly, these two unknown-timepoint recordings support:

- pipeline validation,
- participant-level ERP description,
- hypothesis-constrained model comparison.

They do **not** support population inference, longitudinal recovery claims, or
lesion-aware DCM. Individual MRI, lesion masks, digitized electrodes, known
visit labels, reviewed artifact correction, and a sufficiently sized
longitudinal sample remain necessary for the full study.

## Statistical interpretation

The temporal permutation tests sign-flip trials within one recording after
baseline correction. Significant intervals therefore describe consistent
within-participant trial-level responses. Trials are not substitutes for
participants. These p-values must not be presented as an N=2 population test.

`picture_correct` is a subset of `picture_all`; comparing those two curves as
independent conditions is invalid. They are reported as two requested
descriptions, not as a condition contrast.
