# Mathematical Proofs for T-FAN Validation

## Lemma 1: Coercivity of the Aerodynamic Operator
Let $\mathcal{L}: H^1(\Omega) \to H^{-1}(\Omega)$ be the linearized aerodynamic operator defined by
\[
\langle \mathcal{L}\phi, \psi \rangle = \int_{\Omega} \left( \nabla \phi \cdot \nabla \psi + \beta(x) \phi \psi \right) \mathrm{d}x.
\]
If $\beta(x) \geq \beta_0 > 0$ almost everywhere, then $\mathcal{L}$ is coercive:
\[
\langle \mathcal{L}\phi, \phi \rangle \geq \min(1, \beta_0) \|\phi\|_{H^1(\Omega)}^2.
\]
*Proof.* The bound follows from the Poincaré inequality and the positivity of $\beta(x)$.

## Lemma 2: Compactness of the Turbulence Feedback
Define the feedback operator $\mathcal{K}: H^1(\Omega) \to H^1(\Omega)$ via
\[
(\mathcal{K}\phi)(x) = \int_{\Omega} k(x,y) \phi(y) \, \mathrm{d}y
\]
with kernel $k \in L^2(\Omega \times \Omega)$. The operator is Hilbert-Schmidt, hence compact. This property is used in Section D of the supplement to apply Weyl's inequality.

## Proposition 3: Exponential Stability under Bounded Turbulence
Assume the turbulence intensity parameter satisfies $0 \leq \tau < \lambda / \|\mathcal{K}\|$. Then the energy functional decays exponentially with rate $2(\lambda - \tau \|\mathcal{K}\|)$.

*Proof.* Combine Lemma 1 and Lemma 2 to bound the perturbed generator $\mathcal{T}_\tau = \mathcal{T} + \tau \mathcal{K}$. The standard semigroup estimate for linear operators yields the claimed decay rate.

## Corollary 4: Empirical Validation Bound
For all experiments in dataset TF-2025A with measured turbulence intensity $\tau_{\mathrm{obs}}$, the inequality $\tau_{\mathrm{obs}} < 0.6\, \lambda/\|\mathcal{K}\|$ implies a stability margin of at least $0.4\,\lambda$.

*Proof.* Substitute $\tau = 0.6\, \lambda / \|\mathcal{K}\|$ into Proposition 3.
