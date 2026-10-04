# EVIDENCE_SCOPE — `zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1`

Six corrections about what earlier rounds do and do **not** establish, so that this round is not
read as a continuation of unproven claims. Written before any new dev/target score was produced.

1. **Prior D_g / M_g dictionary-vs-MLP evidence is inconclusive.**
   The completed `zinc_chemistry_dictionary_vs_mlp_seed0_v1` round did not resolve whether the
   dictionary arm (D_g) beats the MLP bridge arm (M_g) on the chemical target; its own report
   classifies the comparison as inconclusive / not separable at the tested budget. Nothing in this
   round may cite D_g > M_g (or the reverse) as established. Reference B here is the M_g raw soup;
   "J/I better than B" is an exploratory performance signal against one historical soup, not a
   claim that local dictionaries beat MLPs in general.

2. **The ring-relief / cycle-relief narrative is unconfirmed.**
   Earlier notes suggested that dictionary structure relieves cycle/ring-related error. The
   completed paired round did not confirm a ring-specific relief effect (group-level
   contributions were not separable at this budget), and the old cycle-label pipeline has the
   provenance caveats below. This round reports per-`k` contributions but makes **no** ring-relief
   claim; `k` groups remain descriptive only.

3. **Earlier rounds contain fit/dev mix-ups.**
   Some prior write-ups quoted a number computed on fit as if it were dev (or vice versa), and
   older summaries propagated the mix-up. Consequences honoured here: (a) all gate statistics are
   recomputed on dev with explicit fit/dev labels in every table; (b) reference B's bias is
   recomputed from `g_fit − raw_soup_fit` and checked against the published fit/dev/G0 numbers;
   (c) old fit-only numbers from previous reports are never reused as dev evidence.

4. **The old structure-binding operator is not the new tuple operator.**
   Previous rounds bound "structure" through the common-subspace / correlation-style dictionary
   machinery (`cssd.CommonSubspace`, `X175` codes, learned/kappa-scaled bindings, kernelised
   patch features). The present operator is an **explicit incidence-count correspondence**
   `(phi_v, a_v, a, t)` with real vs marginal weights. Numerical or qualitative agreement between
   the two families is not assumed; no old binding code path is reused inside the arms.

5. **`X175` input and the new tuple input are different objects.**
   The old feature vector `X175` (with the old binding and normalisation) is not the new
   `x = [phi65; onehot28(a_v); onehot28(a); onehot4(t)]` (125-D). The new tuple index, scaler and
   kappa are built from scratch on the frozen fold; no coordinate or block correspondence with
   `X175` is claimed, and no `X175`-based result is carried into this round's gates.

6. **There is no message passing in the skeleton.**
   The deployed body has no MPNN / propagation layers. Each root sees only its own descriptor and
   its incident tuples (and Sem110 + size2 via the body). Therefore this round tests a *static,
   one-hop, root-local* correspondence question; positive joint-vs-independent evidence would speak
   to local chemistry representation, not to propagation or global structure. No global-structure
   claim follows from either outcome.

**Scope of the round**: exactly two arms (J, I), one seed, one fold, 240 epochs each; reference B
read-only; dev-only analysis; no official-valid/test access; no ring head; no 10k confirmation.
