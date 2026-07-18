# CVLR-XVLA Mathematical Mechanism Audit

## Scope and variables

For each synchronized clean demonstration record, frozen X-VLA/Florence2 produces agent tokens `A in R^(B x 50 x 1024)` and wrist targets `W in R^(B x 50 x 1024)`. Frozen language input embeddings are averaged to `l in R^(B x 1024)`, and the existing X-VLA proprioceptive vector is `p in R^(B x 20)`.

The deployment missing indicator is `m in {0,1}`. It is derived only from the declared wrist-dropout condition and exposes no reward, success, future action, simulator state, or expert action.

## Predictor

The trainable predictor has bottleneck width `128`:

`h_i = GELU(P_A(LN(A_i)) + P_L(l) + P_P(p) + e_i)`

`u_i = GELU(P_C(h_i))`

`W_hat_i = 3 * tanh(P_O(u_i))`

where `e_i` is a learned per-token positional vector. The output projection is initialized exactly to zero, so the initial missing-view prediction is the zero-fill ablation. The module has exactly `422,144` trainable parameters. X-VLA has no trainable parameters.

## Training objective and gradient path

The sole objective is latent reconstruction:

`L_recon = mean((W_hat - stopgrad(W))^2)`

Both `A` and `W` are cached from the frozen visual encoder. Gradients flow only through the CVLR predictor. There is no action loss, auxiliary loss, KL term, contrastive term, RL reward, expert-action supervision, or X-VLA gradient path.

The units are squared Florence2 feature units. The feasibility probe measured target element standard deviation `0.9943..0.9991`, so an unnormalized mean-squared objective has a natural order-one scale. AdamW learning rate `1e-3`, batch size `4`, gradient clipping `1.0`, and exactly `96` optimizer steps are frozen before outcomes.

## Policy integration

Official X-VLA constructs `aux_visual_inputs in R^(B x 100 x 1024)` from two auxiliary view slots. The first `50` tokens exactly equal the wrist-view Florence2 block. A forward pre-hook on `model.transformer` replaces only `aux_visual_inputs[:,0:50,:]` when `m=1`:

`Aux_out = Aux_original` when `m=0`

`Aux_out[:,0:50,:] = W_hat` when `m=1`

All other auxiliary tokens, the language/primary-view encoder output, proprioception, diffusion state, action transformer, and action postprocessing remain unchanged. Clean bypass returns the original tensor without cloning or arithmetic, permitting bit-exact action passthrough.

## Expected behavioral effect

Under wrist dropout, zero or black wrist features remove close-range hand-object evidence. Reconstructing the wrist token block should restore part of the representation expected by the frozen X-VLA action transformer. The falsifiable chain is:

`agent/language/proprio -> lower wrist-latent validation error -> changed dropout action output -> bounded Stage A benefit`

Stage 0 tests the first two arrows only. It makes no closed-loop success claim.

## Comparators and alternatives

- `ZERO_FILL / CVLR_NO_RECONSTRUCTION`: all-zero wrist token block; it is both the key ablation and missing-latent baseline.
- `AWF_DETERMINISTIC_AGENT_TOKEN_FILL`: copy current agent-view tokens into the wrist block; it tests whether a deterministic no-training substitution explains reconstruction or action effects.
- Frozen dropout Base retains X-VLA's existing black-pixel wrist encoding and anchors action disruption metrics.
- RL4IL remains the closest locally validated external prior for a later closed-loop comparison; it is not recast as VLA training.

Standard LoRA is omitted because X-VLA stays frozen and generic policy adaptation does not test cross-view wrist-latent reconstruction.

## External-prior distinction

WristWorld targets the same anchor-to-wrist gap but reconstructs geometry and synthesizes pixel videos for downstream data augmentation. CVLR predicts the existing policy's wrist latent block directly and intervenes only for a missing live view. MV-MWM reconstructs masked pixels for representation/world-model learning; RPT predicts generic masked sensorimotor latents; ReconVLA reconstructs gaze-region tokens for grounding. None is the exact frozen-X-VLA missing-wrist token replacement defined here.

## Safety and identity audit

Continuous translation, continuous rotation, raw pre-discretization gripper score, and final binary gripper changes are measured separately. No universal max-absolute threshold combines them. Clean actions must be exact passthrough. Dropout translation/rotation and raw gripper changes are bounded independently, with at most one discrete gripper flip across the nine live dropout rows.

Training demos `0/20` and validation demo `40` are disjoint within the standing `0..39 / 40..49` split. Action probes use only the frozen development identities. No confirmatory test identity, outcome-based checkpoint selection, natural-reset mining, or privileged inference input is permitted.

## Known failure modes and decisions

- Invalid or collapsed targets: `DATA_OR_SUPERVISION_FAILURE`.
- Missing gradients, wrong insertion, wrong checkpoint, or no real X-VLA/CUDA path: `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.
- Valid but non-improving reconstruction or action-indistinguishable output: `KEY_COMPONENT_NOT_USEFUL`.
- Clean-bypass or semantic-aware safety failure: `DESIGN_FAILURE`.
- Only a complete Stage 0 pass may trigger a separately frozen bounded Stage A.
