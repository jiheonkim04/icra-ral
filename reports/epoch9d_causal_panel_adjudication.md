# Epoch 9D Exact-State Mass-Swap Causal Adjudication

Decision: `CAUSAL_SIGNAL_GO`

Paper: `PAPER_NOT_AUTHORIZED`

The sealed panel completed 32/32 primary assignments, 64/64 candidate probes, and 16/16 sham rows. All rows, including failures, are retained in `reports/epoch9d_causal_panel/result.json`.

## Primary counts

- finite bounded actions: 64/64
- intended contact or excitation: 64/64
- both candidates excited: 32/32
- correct heavy/light ranking: 28/32
- front-heavy ranking: 15/16
- back-heavy ranking: 13/16
- exact pairs with both assignments correctly flipped: 12/16
- collisions / identity swaps / falls / workspace exits / unrecoverable track losses: 0 / 0 / 0 / 0 / 0

## Paired causal effect

The preregistered exact-pair contrast (back-light response minus back-heavy response) is `0.006593 m` on average. Its paired 95% Student-t interval is `[0.004074, 0.009113] m`; 15/16 nonzero pairs have the expected sign, giving one-sided exact sign-test `p = 0.000259399`. The centered position/lane/order HC3 estimate is `0.006593 m` with interval `[0.003396, 0.009791] m`.

## Shortcut and sham controls

Exact paired first RGB matched in 16/16 pairs and initial localization fields matched in 16/16. A deterministic position/order/pre-contact-only rule cannot flip within an exact pair and therefore scores exactly 16/32 at best under swapped labels. The sham ran 8 paired base states with 0 sampled contacts, 0 collisions, and 0 prediction flips; its paired contrast interval is `[-0.000002, 0.000001] m`.

## Gate record

`CAUSAL_SIGNAL_GO` is true. Near-miss replication eligibility is false. Every individual gate and exact-pair row is recorded in `reports/epoch9d_causal_panel_adjudication.json`. No validation or confirmation identity was accessed.
