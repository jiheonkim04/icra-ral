# Epoch 8 Validation Language-Pair Audit

Status: `COMPLETE_BEFORE_CANDIDATE_FORMULATION`

This audit was completed from instruction text and frozen benchmark metadata only. No model, simulator, or Ours outcome was loaded. All 30 final validation pairs were reviewed before candidate formulation.

## Criteria

- The wording must be grammatical enough to determine an executable request.
- The request or pragmatically licensed hint must entail the canonical task goal in the released benchmark scene.
- Object aliases may vary, but may not introduce a plausible different physical referent or state predicate.
- No extra action, target, or intent may change the benchmark success condition.

## Summary

- Final accepted pairs: 30 / 30
- Rejected ranked attempts: 3
- Reviewed attempts: 33
- Selection after rejection: advance to the next row under the original frozen SHA256 ranking; no manual replacement choice.

## Decisions

| Task | Family | Decision | Row ID | Paraphrase | Reason |
|---:|---|---|---|---|---|
| 0 | act | ACCEPT | `6ca68453eded7fa301a384c57f63cde5907bdf57172e350237e15b90fecc1645` | pull the middle drawer of the cabinet | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 0 | obj | ACCEPT | `6d74ebb17b5157b83888c92dd89c75889f571c9d55420c9b4a70702c9b6729da` | open the middle drawer of the chest | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 0 | comp | ACCEPT | `f572a394e29e9dfbe5c4717b588afeaa474f9029973959d67723867874bc3a99` | pull the middle bin of the cupboard | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 1 | act | ACCEPT | `ece35e74831d2bedf009ea8340b3476dfff2c7915ca95ce4d4d0dff8c60edbb7` | Can you open the top drawer and put the bowl inside? | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 1 | obj | ACCEPT | `e18a3787f3cee18457be7603add6f331801a44eed3cdc4a23e9265aeb0ca6e22` | open the top drawer and put the soup bowl inside | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 1 | comp | ACCEPT | `602e99b2fa39ad65bccdddc12bb239e681a841fcc4e3aa11e264acf149d5921a` | Is the basin meant to go in the top receptacle? | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 2 | act | ACCEPT | `d2ca56e627811eb0b061ed72001af980cd30f7a87a6d415ad90628e79e1806dd` | I thought the plate would be at the front of the stove | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 2 | obj | ACCEPT | `21a36799af56ca930521a053cd59bdc24ce2074cf0b631135919b47a3552493c` | push the platter to the front of the stovetop | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 2 | comp | ACCEPT | `68de464f8de61e80cdd61509f2d16bb5cc172f0d7f9a184461a4a1265cc4898a` | slide the platter to the front of the range | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 2 | comp | REJECT | `4a74defd2da3967a4722ac16874ef037eb7201943213e51c40f4771fcbc00b1b` | Will that saucer be used from the front of the cooking surface? | The question does not entail pushing or moving the plate to the requested goal state. |
| 3 | act | ACCEPT | `3491dfaca57bbb2b133a51c8876690b7370a78b80c8814a5afb10a061fa6cc36` | Could you put the bowl on the plate? | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 3 | obj | ACCEPT | `7a4f7c77f0978bcb1e5f9e5c6023a83bb21f50d72aaffa59d5d7dd2df67a121e` | put the bowl on the side plate | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 3 | comp | ACCEPT | `a3c7a10d32002b52294aaa62fb8df39bced3babbef12355f00cc2e492bb1d0a4` | arrange the bowl on the platter | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 4 | act | ACCEPT | `99b061087d862a68e5a2973b592f29ec6ed16a2b857dcb33f203464d73667027` | The bowl goes on the stove | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 4 | obj | ACCEPT | `6e47cbc0b3cef547e8795b4fa5e510f064e6cf1100ab335af7a06b0fc1ac3ec5` | put the mixing bowl on the stove | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 4 | comp | ACCEPT | `2d95e99d23cf6c9a771a1eb06c943c01330450facd2a763bd11b9ab1eb048940` | get the receptacle and then place it on the range | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 4 | comp | REJECT | `8c1113dab269a9ae412d4ae10d9cc9ea9fcec8f065874156b212b74edd88f592` | set the pot on the burner | The paraphrase substitutes pot for bowl and burner for stove, so the physical referents are not reliably equivalent. |
| 4 | comp | REJECT | `ba14d4a6c9307a8e40136c9b9f4b2c6ab434c8622dfabf727072db88f0504e10` | Is the pan meant to be heated on the cooking surface? | The paraphrase adds a heating intent and substitutes pan for bowl, changing both object identity and requested state. |
| 5 | act | ACCEPT | `64630a19d093f5bf2e0c0becf967878ac43d97adcff2535eac59a0004e427fb0` | Is the top of the cabinet the correct place for the bowl? | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 5 | obj | ACCEPT | `b9e7e036f93c0525f2c8d6eafef5c748a3e50be46cc5edbcf0953c5ee6a5d278` | put the fruit bowl on top of the cabinet | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 5 | comp | ACCEPT | `b124a1ed6cbd52834d810c29c4fe4e67e5a73e0a97697e0cc7668e8e00412a15` | Get the cereal bowl and then put it on top of the display cabinet | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 6 | act | ACCEPT | `abae7fedca97137157bee27153de779d29b79b1d8fc34bef7b6148924a7bf5e6` | I need to have the cream cheese in the bowl | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 6 | obj | ACCEPT | `e6a03935ae5a0831ccec7ac684dd5295f08dd207484de5ff571a659f41c2c84b` | put the cheese in the serving bowl | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 6 | comp | ACCEPT | `5bb2db362b1a5c2cb3c022b1ebda47412b6dd692ef4ddd589a77ce9cf870e930` | The prep bowl is ready for the cheese | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 7 | act | ACCEPT | `f624ae9074031d79770c087cec76f6250809fef43998fa02bb127d0c358902b8` | Could I get the wine bottle onto the rack? | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 7 | obj | ACCEPT | `68a4f8969af44905af0b4551eb4c39d9bdc3d785bab4c9738a500c97c5693259` | put the wine bottle on the storage rack | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 7 | comp | ACCEPT | `b2842bbc0df1903f9e270af4c603e87cda44a446ac66a00996228aa612e728d3` | Could I have the wine vessel put on the stand? | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 8 | act | ACCEPT | `cdb06a437f81019116251c9d96387b1eb3597b4ac51317d1ecc00024cb5e177e` | Does the wine bottle belong on top of the cabinet? | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 8 | obj | ACCEPT | `af71b66dcbff982b33920d06840370d6792af73f2be04e173e921afd39f25bff` | put the drink bottle on top of the hutch | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 8 | comp | ACCEPT | `128e35d675bbcefb6b823789fa4b199c09f7814a9ee07f85628ed0b684118b48` | The top of the storage cabinet is clear for the drink bottle | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 9 | act | ACCEPT | `c228bffcd5a8dcfb16adae2ba6b5ece0c76f31313ce2fecdfb525187ed0b0d28` | The stove needs to be on now | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 9 | obj | ACCEPT | `24e09b137e2ea0474ea99854fd1bdc23b2f4008ae13cf29ab247df80a97eb013` | turn on the cooker | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |
| 9 | comp | ACCEPT | `076f31badb1a1a98bfe863333831bb540a68ade3dc5b409f97b84cdbdece00de` | Could you please turn on the range? | Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition. |

## Leakage Check

`ours_outcomes_observed=false`; `model_loaded=false`; `simulator_episode_count=0`. Confirmation language text remains sealed.
