# Official SmolVLA Evaluation Determinism Check

Date: 2026-07-10 KST

Performed: `True`
Scope: `val+test fixed manifest; no training`
Eval seed policy: `Before each disk-evaluation pass this audit sets torch.manual_seed(seed) and np.random.seed(seed). This proves fixed-seed repeatability and exposes that unpinned evaluation RNG state is part of the protocol identity.`
Repeat tolerance: `1e-06`
All deterministic within tolerance: `True`
Saved regenerated artifact matches fixed-seed re-eval: `False`

| seed | max action abs diff | max action L2 diff | rank4 metric diff | static metric diff | alpha identical | static metric identical | deterministic |
| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 11 | 0.0 | 0.0 | 0.0 | 0.0 | `True` | `True` | `True` |
| 22 | 0.0 | 0.0 | 0.0 | 0.0 | `True` | `True` | `True` |
| 33 | 0.0 | 0.0 | 0.0 | 0.0 | `True` | `True` | `True` |

## Pass Metrics

### Seed 11
- pass `1`: rank4 `0.087583654`, static `0.079568273`, alpha `0.5`, digest `253255C989773FF37E207BCA62278D3E60E2BAC46852F9B3FD64595249ABC715`
- pass `2`: rank4 `0.087583654`, static `0.079568273`, alpha `0.5`, digest `253255C989773FF37E207BCA62278D3E60E2BAC46852F9B3FD64595249ABC715`
### Seed 22
- pass `1`: rank4 `0.086454128`, static `0.078693054`, alpha `0.5`, digest `48849FF18D56F7B2EFEB824C433415B5A05982CE518FEE009C6B3613F3A35711`
- pass `2`: rank4 `0.086454128`, static `0.078693054`, alpha `0.5`, digest `48849FF18D56F7B2EFEB824C433415B5A05982CE518FEE009C6B3613F3A35711`
### Seed 33
- pass `1`: rank4 `0.084401121`, static `0.07804313`, alpha `0.5`, digest `5A1E454FF96341D0749D0E3104F5C6E6879A9CE3C704B882A5BF88E68C808608`
- pass `2`: rank4 `0.084401121`, static `0.07804313`, alpha `0.5`, digest `5A1E454FF96341D0749D0E3104F5C6E6879A9CE3C704B882A5BF88E68C808608`

## Saved Regenerated Artifact Vs Fixed-Seed Re-Eval

| seed | regenerated rank4 | fixed-seed rank4 | diff | regenerated static | fixed-seed static | diff | regenerated alpha | fixed-seed alpha | match |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 11 | 0.087213856 | 0.087583654 | 0.000369798 | 0.078911385 | 0.079568273 | 0.000656888 | 0.5 | 0.5 | `False` |
| 22 | 0.088713382 | 0.086454128 | 0.002259254 | 0.080982228 | 0.078693054 | 0.002289174 | 0.25 | 0.5 | `False` |
| 33 | 0.085934428 | 0.084401121 | 0.001533307 | 0.078716617 | 0.07804313 | 0.000673487 | 0.5 | 0.5 | `False` |
