# Phase 2 CPU gate record

- Round: `zinc-edit-conditioned-local-contribution-v1`
- Created: `2026-10-29T22:38:35+0800`
- All pass: **`True`**

| check | pass | key evidence |
|---|:---:|---|
| C1_independent_reference | yes | n_molecules=14; n_atoms=332; max_abs_diff=0.0; max_int_diff=0 |
| C2_synthetic_radius1_invariance | yes | radius1_block_identical=True; radius2_block_changed=True; real_radius2_input_zero=True; real_fbase_target_exactly_equal=True |
| C3_invariance | yes | max_abs_atom_renumber=4.76837158203125e-07; max_abs_bond_direction=2.384185791015625e-07; max_abs_batch_split=5.960464477539063e-08; max_abs_molecule_order=0.0 |
| C4_parameter_identity | yes | parameters=248368; breakdown={'atom_encoder': 26336, 'bond_encoder': 62816, 'local': 133645, 'f_context': 4225, 'rho': 10497, 'f_base': 10849, 'backbone': 237519, 'total': 248368}; cap=500000; same_count=True |
| C5_initial_lcrc_equivalence | yes | max_abs_real_vs_lcrcA=0.0; max_abs_null_vs_lcrcA=0.0; shared_backbone_tensors_equal=True |
| C6_gradients_and_paths | yes | arms={'REAL': {'step1': {'loss_mol': 1.6053695678710938, 'loss_pair': 0.6764082312583923, 'pair_target_std': 1.0494929552078247, 'base_last_grad_norm': 1.3297855854034424, 'base_first_grad_norm': 0.0, 'base_r2_col_grad_max_abs': 0.0, 'base_r1_col_grad_max_abs': 0.0, 'f_context_grad_norm': 21.843324661254883, 'rho_grad_norm': 0.9502238035202026, 'all_finite': True}, 'step2': {'base_r1_col_grad_max_abs': 0.009361271746456623, 'base_r2_col_grad_max_abs': 0.0, 'base_last_grad_norm': 2.9746973514556885}, 'pair_only': {'base_last_grad_norm': 0.6290050148963928, 'f_context_grad_norm': 11.050552368164062}, 'pass': True}, 'NULL': {'step1': {'loss_mol': 1.6053695678710938, 'loss_pair': 0.6764082312583923, 'pair_target_std': 1.0494929552078247, 'base_last_grad_norm': 1.5456089973449707, 'base_first_grad_norm': 0.0, 'base_r2_col_grad_max_abs': 0.0, 'base_r1_col_grad_max_abs': 0.0, 'f_context_grad_norm': 21.843324661254883, 'rho_grad_norm': 0.9502238035202026, 'all_finite': True}, 'step2': {'base_r1_col_grad_max_abs': 0.009487401694059372, 'base_r2_col_grad_max_abs': 0.011544504202902317, 'base_last_grad_norm': 3.4902751445770264}, 'pair_only': {'base_last_grad_norm': 0.8325825929641724, 'f_context_grad_norm': 11.050552368164062}, 'pass': True}} |
| C7_checkpoint_roundtrip | yes | arms={'REAL': {'max_abs_diff': 0.0}, 'NULL': {'max_abs_diff': 0.0}}; tolerance=1e-06 |
| C8_split_isolation | yes | pair_rows_in_train=True; no_self_pairs=True; code_refs_to_lc_test=[]; valid_rows=1000 |
| C9_pair_loss_algebra | yes | additive_identity_abs_diff=1.1920928955078125e-07; pair_loss_offset_diff=0.0; molecule_loss_offset_delta=3.6999998092651367; pair_vs_molecule_grad_cosine=-0.215591162443161 |
| C10_mechanism_difference | yes | max_abs_fbase_diff=0.3783910870552063; max_abs_fcontext_diff=0.0; max_abs_full_pred_diff=6.0256452560424805; fraction_atoms_with_radius2_content=1.0 |
| C11_post_step_non_equivalence | yes | arms={'REAL': {'max_abs_vs_lcrcA': 25.271127700805664, 'max_abs_zero_base_ablation': 0.04535484313964844, 'pass': True}, 'NULL': {'max_abs_vs_lcrcA': 25.274564743041992, 'max_abs_zero_base_ablation': 0.04879188537597656, 'pass': True}} |
| C12_streams_identical | yes | data_order_first_1000_sha=2631760808; pair_stream_first_1000_sha=3290709025 |
