# Deep V3 core adaptations (pre-registered, no performance-based tuning)

These are local self-contained **core adaptations**, not exact official reproductions.
Pinned source commits, reference file SHA256s and configuration references are in
`provenance.json`. The original repositories are not runtime dependencies.

## Inputs, outputs and sample independence

All eight baselines use the existing deterministic neutral adapter unchanged:
stock mean/population std, bond mean/population std, then 24 commodity nodes.
RNN/GRU/LSTM/Transformer flatten token-major then feature-major to B,20,588.
Graph WaveNet and existing adapted MTGNN retain B,20,28,21.
MSGNet and CrossGNN each own a shared trainable Linear(21,1); all 21 features
enter this projection for every node. They never treat 588 features as economic nodes.
All outputs use four supervised slots [1,5,10,20], not four contiguous future days.
Readout rows 4:28 map to the verified commodity target order.

Upstream MSGNet and CrossGNN select FFT peaks after **averaging amplitudes across
batch samples**. This conflicts with this benchmark's single-sample independence
requirement. Both local cores select peaks separately for each sample (channel
mean only), excluding DC explicitly. Stable ties use ascending frequency index.
Grouping equal periods merely batches the same operation; it cannot alter a
sample's selected periods or predictions. This is an explicit methodological
adaptation, not a claim of verbatim reproduction.

## Graph WaveNet

Reference: nnzhan/Graph-WaveNet, pinned official core already recorded in the
project's third_party provenance. Local `graph_wavenet.py` retains eight dilated
gated convolutions (four blocks, dilation1/2, kernel2), order2 graph diffusion,
residual/skip paths and 256→512→4 output convolutions. Adaptive-only adjacency
softmax(ReLU(E1 E2)), embedding10; one A support; no fixed/physical/D0B adjacency.
Every convolution is left padded, length preserving; receptive field13.

Official valid convolutions/cropped residuals are expressed as left-padded aligned
states so prefix causality is directly auditable. The final valid timestep has
the same receptive-field coverage. Official BatchNorm over batch/node/time is
omitted: it would mix samples and future positions during training, contrary to
the strict causal/sample-independent path requested here. No replacement learned
normalizer is added. No unused residual projection for a graph-disabled mode is
created. Dropout=.1, input21, node28, output4 and dimensions follow the task.
The earlier vendored full-node implementation is untouched.

## MSGNet (AAAI 2024)

References: `YoZhibo/MSGNet/models/MSGNet.py`, `layers/MSGBlock.py`,
`layers/Embed.py`; small ETTh1 script and `run_longExp.py` defaults. Local files
preserve value/absolute-position embedding, FFT top3 scales, three independent
adaptive graph embeddings, depth2 teleport MixHop (alpha=.3), graph residual/LN,
period reshaping, 8-head intra-scale attention/FFN, frequency-softmax scale
aggregation and encoder residual. Dimensions match the fixed request.

The upstream graph block maps latent32→node28 using a (5,1) convolution, applies
MixHop and maps back with a (1,20) convolution plus Linear(28,32). That mapping is
retained; only squeeze(-1) is used so a singleton batch cannot disappear.
The embedding uses the official circular kernel3 value convolution and sinusoidal
absolute positions, without unavailable timestamp covariates. This operates only
inside the fully observed window and is **not** a prefix-causal encoder.

The protocol requires each scale to be a parallel graph/attention path. Therefore
paths all start from the same embedded window, each with its own attention;
upstream's sequential overwrite of x and shared attention are not copied.
Scale aggregation adds the original embedded window. Official Q/K/V/output
attention is expressed with PyTorch MultiheadAttention; its within-period causal
mask and residual/LN/GELU/FFN/dropout structure remain. FFT is on the embedded
observed sequence as in the reference.

No extra input-window standardization and **no input-mean/std output restoration**
are applied: the existing model-ready data are kept, and learned node scalars
are not target returns. After projection32→28, a shared node-wise Linear(20,4)
with dropout.1 predicts horizon slots. Output restoration would mix physical
quantities. The learned 21→1 adapter and non-contiguous target slots are task
adaptations. No scale/graph auxiliary loss.

## CrossGNN (NeurIPS 2023)

References: `hqh0728/CrossGNN/models/CrossGNN.py` and `run_longExp.py` defaults.
Local `crossgnn.py` retains original scale1 plus four FFT-identified periods
(ceil(T/frequency)), stride=period average-pooling, concatenate/pad/crop to2T,
scalar→hidden8, learned timevec1/timevec2 graph and order2 Cross-Scale GNN,
learned nodevec1/nodevec2 signed variable graph and order2 Cross-Variable GNN,
both graph residual refinements, original/refined channel concatenation,
Linear16→1, and shared Linear(20,4) output. One graph encoder, dropout.05.

Temporal graph saliency uses top max(tk//period,5), clipped to each scale's length,
plus local predecessor/successor links. The local code uses explicit two-index
adjacency assignment, fixing the upstream advanced-indexing ambiguity that could
unmask entire rows. It retains source-column normalization. The variable graph
retains top3 positive softmax(score) and bottom3 negative softmax(1/(score+1)),
including threshold ties. Unused graph-embedding and normalization modules for
disabled branches are not instantiated. No D0B graph component is reused.
All parameters construct device-agnostically; temporary tensors follow input or
parameter device/dtype. No hard-coded CUDA index or unregistered .to() Parameter.
`anti_ood=False`: input last-value restoration has no target-return interpretation.
All scales use observed history only; this encoder is not prefix causal.

## Existing definitions and training

Neutral adapter, GRU, VanillaTransformer and adapted MTGNN classes are unchanged.
RNN/LSTM differ from GRU only in the requested standard recurrent cell.
`baseline_protocol.train_one`, `prediction_loss`, validation selection and
population_metrics are reused. No auxiliary loss, tuning, clipping, AMP,
accumulation or reduced batch. OOM/nonfinite/sanity failure stops the suite.
