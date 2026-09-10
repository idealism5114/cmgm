# Official baseline provenance and adaptations

Pinned official repositories, commits and per-file upstream/local SHA256 values are in baseline_provenance.json. Both authors' MIT licenses and README files are retained. No simplified graph architecture is used.

## Graph WaveNet

Source: https://github.com/nnzhan/Graph-WaveNet at 6b162e80c59a1d494809252eca055cff93dc66b1.

The three `nn.Conv1d` constructors in model.py (gate, residual, skip) receive four-dimensional NCHW tensors and two-element kernels in upstream code. Modern PyTorch rejects these inputs. They are changed to `nn.Conv2d`; weight/bias shapes and the intended convolution computation remain identical. model.py.patch records the exact diff. No gates, graph convolution, skip/residual paths, BatchNorm or layers are removed.

Official defaults retained: blocks4, layers2, kernel2, residual/dilation32, skip256, end512, dropout.3; adaptive-only graph with no fixed support. Input is losslessly permuted B,T,N,F → B,F,N,T, in_dim21, out_dim4, num_nodes from dataset. Upstream outputs B,4,N,L; read the final forecast-origin column and dataset commodity indices. The unchanged official receptive field is13: all20 steps are supplied, but the final output has a13-step effective receptive field. This architectural limitation is disclosed, not hidden by claiming every timestep has a nonzero output dependency.

## MTGNN

Source: https://github.com/nnzhan/MTGNN at f811746fa7022ebf336f9ecd2434af5f365ecbf6.

Only net.py import changes from `from layer import *` to `from .layer import *`; net.py.patch records it. layer.py is byte-identical to upstream. Defaults match official train_multi_step.py: gcn/buildA true, depth2, dropout.3, subgraph20, node_dim40, dilation_exponential1, conv/residual32, skip64, end128, layers3, propalpha.05, tanhalpha3, layer_norm_affline true. No subgraph sampling/node deletion. seq_length20, in_dim21, out_dim4 and dataset num_nodes are interface configuration changes. Input B,F,N,T, output B,4,N,1; select commodity indices and squeeze last singleton dimension.

Official graph construction uses random perturbations before top-k even in eval. This is preserved; batch/single-sample tests compare the same RNG realization. Official LayerNorm spans channel/node/time of the observed window: the convolution paths are causal, but intermediate post-normalization prefix states need not be prefix-invariant. We test convolution right-edge alignment and legal observed-window use; we do not falsely claim the whole MTGNN internal timeline is strictly streaming-causal. No future forecast feature/target enters either official graph model.

## Common task interface

Wrapper and full-information neural baselines are in cmgm/models/formal_baselines_v2.py. Only the official graph output interface is adapted. No D0B graph, encoder, weights or features are imported into graph baselines. The original official optimization engines are replaced by the requested common Huber/Adam/VAL5-selection protocol for all neural models; graph curriculum learning and original optimizer-specific clipping are not used. These are the predeclared benchmark training policy, not hidden architecture changes.
