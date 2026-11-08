# Unknown-Appliance Detection in NILM with Autoencoders (UK-DALE)

This repository compares three autoencoders — **USAD**, **VAE** and **GRU-AE** — on a **novelty detection** task in non-intrusive load monitoring (NILM): can a model trained only on *known* appliances detect, through its reconstruction error, the appearance of an **unknown appliance** in the aggregate signal?

The three notebooks follow **exactly the same protocol**. Only the model (step 4) changes, which makes the results directly comparable.

## Repository contents

| Notebook | Model | Distinguishing feature |
|---|---|---|
| `usad_nilm_novelty_detection.ipynb` | **USAD** (Audibert et al., KDD 2020) | Shared encoder, two decoders, two-phase adversarial training |
| `vae_nilm_novelty_detection.ipynb` | **VAE** | Probabilistic bottleneck (`mu` / `logvar`, reparameterization, KL term), single decoder |
| `gruae_nilm_novelty_detection.ipynb` | **GRU-AE** | Recurrent (GRU) encoder and decoder, single decoder, plain MSE loss |

The benchmark variable is therefore the **architecture family**: dense with an adversarial dual decoder (USAD), dense probabilistic (VAE), recurrent (GRU-AE).

## Protocol

1. **Data**: UK-DALE, House 1, read with [NILMTK](https://github.com/nilmtk/nilmtk).
2. **Synthetic aggregate**: sum of the *ground-truth* power (submeters) of the known appliances only — never the real house mains.
3. **Windowing**: sliding windows (`WINDOW_SIZE = 200` samples, `STEP = 5`, 6 s sampling period), z-score normalization fitted on the training windows.
4. **Training**: autoencoder trained on "known appliances" windows only (50 epochs, batch size 1024, latent dimension 14).
5. **Anomaly score**: per-window reconstruction error.
   - USAD: `0.3 · MSE(w1) + 0.7 · MSE(w2)`
   - VAE: reconstruction MSE, decoded deterministically from `mu` (the KL term is excluded from the score)
   - GRU-AE: reconstruction MSE
6. **Progressive test**: inject one novel appliance at a time into a test period that is disjoint from the training period, then compute ROC-AUC, PR-AUC and two operating thresholds.

### Default configuration

| Parameter | Value |
|---|---|
| Known appliances | `fridge freezer`, `washing machine`, `dish washer` |
| "Low-power" novel appliance | `microwave` |
| "High-power" novel appliance | `kettle` |
| Training period | 2013-04-01 → 2014-05-31 |
| Test period | 2014-07-01 → 2014-08-31 |
| Validation fraction | 15% |
| Random seed | 42 |

The training and test periods are deliberately **disjoint** to avoid any data leakage.

### Decision thresholds

- **Threshold 1 (ROC / Youden's J)**: the point on the ROC curve maximizing `TPR − FPR`, as in the USAD paper.
- **Threshold 2 (F1-optimal)**: the point on the precision-recall curve maximizing the F1-score.

Both are computed on the labeled test windows, with all novel-appliance levels pooled together.

### Sequential test by period

The test period is split into three equal sub-periods: (1) known aggregate only, (2) + low-power appliance, (3) + high-power appliance. The notebooks plot the signal in watts with the zones detected as anomalous, along with detailed diagnostics (score, state of the injected appliance, input vs. reconstruction).

## Results (reference run)

**ROC-AUC / PR-AUC per novel appliance** (about 6% positive windows):

| Model | Microwave (low) ROC-AUC | Microwave PR-AUC | Kettle (high) ROC-AUC | Kettle PR-AUC | Pooled ROC-AUC |
|---|---|---|---|---|---|
| USAD | 0.964 | 0.457 | 0.977 | 0.509 | 0.970 |
| VAE | 0.972 | 0.573 | **0.989** | **0.815** | **0.981** |
| GRU-AE | **0.978** | **0.659** | 0.985 | 0.763 | **0.981** |

**Operating points on the pooled test data:**

| Model | Threshold 1: FPR | Threshold 1: TPR | Threshold 2: FPR | Threshold 2: TPR | Threshold 2: F1 |
|---|---|---|---|---|---|
| USAD | 0.054 | 0.986 | 0.036 | 0.924 | 0.743 |
| VAE | 0.079 | 0.989 | 0.038 | 0.876 | 0.711 |
| GRU-AE | 0.098 | 0.995 | 0.026 | 0.808 | 0.730 |

All three models detect the novel appliance well (ROC-AUC > 0.96). The "low-power" case is the hardest, as expected. PR-AUC remains modest because only ~6% of windows are positive: it should be read against that base rate.

> A new run may give slightly different values (GPU non-determinism).

## Installation

### Requirements

- Python 3.9+
- UK-DALE converted to HDF5 for NILMTK (`ukdale.h5`)
- PyTorch, NumPy, pandas, scikit-learn, matplotlib
- `nilmtk` and `nilm_metadata`, installed from GitHub (without their pinned dependencies):

```bash
pip install -q --no-deps git+https://github.com/nilmtk/nilm_metadata.git
pip install -q --no-deps git+https://github.com/nilmtk/nilmtk.git
```

A GPU speeds up training but is optional.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `UKDALE_H5_PATH` | Path to `ukdale.h5` | Kaggle path if present, otherwise `data/ukdale.h5` |
| `OUTPUT_DIR` | Output folder (models, figures) | `/kaggle/working` on Kaggle, otherwise `outputs/` |

### Running

```bash
export UKDALE_H5_PATH=/path/to/ukdale.h5
jupyter notebook
```

Open and run any of the three notebooks from top to bottom. The trained weights are saved in `OUTPUT_DIR` (`usad_nilm_model.pth`, `vae_nilm_model.pth`, `gruae_nilm_model.pth`).

**On Kaggle**: attach the UK-DALE dataset, turn Internet on (to install `nilmtk`) and, if desired, enable a GPU.

## Limitations to keep in mind

- **Thresholds are chosen on labeled test data.** ROC-AUC and PR-AUC do not depend on a threshold, but FPR, TPR, F1 and accuracy at thresholds 1 and 2 are **optimistic**. A more rigorous protocol would select the threshold on a separate labeled validation period.
- **Validation is not independent.** The train/validation split is a random permutation of heavily overlapping sliding windows (200 samples, stride 5): the validation loss is therefore not a faithful estimate of generalization.
- **Narrow setting.** A single house, a synthetic aggregate of three known appliances, only two novel appliances, and hand-chosen ON thresholds (fridge freezer 50 W, washing machine 20 W, dish washer 20 W, kettle 2000 W, microwave 200 W).
- **Single run per model.** One seed, one training run: differences of about 0.01 ROC-AUC between models may not be significant.
- **VAE particularity.** In `reparameterize`, the standard deviation is computed with `exp(0.05 · logvar)` instead of the standard `exp(0.5 · logvar)` assumed by the KL term. The results above were obtained with 0.05 (`KL_WEIGHT = 0.01`); switching to 0.5 gives a standard VAE and requires re-running the experiment.

## Possible improvements

- Select thresholds on a dedicated validation period
- Repeat the experiments over several seeds and several UK-DALE houses
- Add more novel appliances and other combinations of known appliances
- Split train/validation by time blocks instead of a random permutation

## References

- Audibert, J., Michiardi, P., Guyard, F., Marti, S., Zuluaga, M. A. **USAD: UnSupervised Anomaly Detection on Multivariate Time Series**. KDD 2020.
- Kelly, J., Knottenbelt, W. **The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes**. Scientific Data, 2015.
- Batra, N. et al. **NILMTK: An Open Source Toolkit for Non-intrusive Load Monitoring**. e-Energy 2014.

## License

This project is distributed under the MIT license. See the [LICENSE](LICENSE) file.

The UK-DALE dataset is not included in this repository and remains subject to its own license.
