# DECISION — e2e_dictenv_sem108_v1

`M_S = 0.123705` → band **SEM108_WITHIN_EXISTING_PERFORMANCE_BAND**.

`G_sem_shuffle = 0.493259` (`strong`), `G_corr = 0.038245` (`clear_incremental`).

**Case D — DICTIONARY_INCREMENT_RETAINED_BUT_NO_NEW_TASK_BAND**

Node-side dictionary correspondence is exactly zero (G_node = 0.0), and the frozen soup state confirms why: W_A_S / W_A_C collapsed to float32 denormals (absmax 7.050e-38 / 7.048e-38) during training, so node slots are identically zero and the node assignment shuffle is a no-op. The path is alive in the 8-epoch smoke state, so this is a learned redundancy collapse (the direct Sem108 interface already carries the atom chemistry), not a coding defect. The pre-registered G_corr = max(G_node, G_edge) therefore reflects the edge correspondence only.

Single seed-0 trajectory, no matched baseline rerun, no official-test read.
Next-step authorization is recorded in the analysis note; no rescue run was executed.
