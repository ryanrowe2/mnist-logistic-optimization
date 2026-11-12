# Mathematical foundations and correctness

## 1. Binary logistic objective

The augmented design matrix has rows \(x_i^\top\); labels are \(y_i\in\{-1,+1\}\). For the improved evaluation protocol, a linear decision function predicts +1 for \(x_i^\top w\geq 0\), otherwise -1. Historical reconstruction instead preserves the original zero-tie output. Define the signed margin \(m_i=y_i x_i^\top w\), and softplus \(s(t)=\log(1+e^t)\). Then

\[
F(w)=\frac1n\sum_{i=1}^n s(-m_i) + \frac{\lambda}{2}\|Pw\|_2^2,
\]

where \(P\) is the identity unless the last coordinate is an unpenalized intercept, in which case its last diagonal is zero. The original 2024 objective takes \(\lambda=0\) and does not use an intercept.

## 2. Derivative and finite-difference verification

\(s'(t)=\sigma(t)\), \(\sigma(t)=1/(1+e^{-t})\). By the chain rule,

\[
\frac{\partial F}{\partial w_j}
= -\frac1n\sum_{i=1}^n y_i x_{ij}\sigma(-m_i) + \lambda (Pw)_j.
\]

Vector form:

\[
\nabla F(w)=-\frac1nX^\top\big(y\odot\sigma(-y\odot Xw)\big)+\lambda Pw.
\]

The implementation's `logistic_gradient` computes this directly in NumPy. Tests estimate each coordinate as \([F(w+\epsilon e_j)-F(w-\epsilon e_j)]/(2\epsilon)\), with \(\epsilon=10^{-6}\), across regularized, unregularized, and unpenalized-intercept variants. Both numerical and symbolic checks are in `tests/test_gradient.py`.

## 3. Stable arithmetic

The historical notebook computes `np.log(1 + np.exp(-margin))`, which overflows for sufficiently negative margins. Its gradient computes `np.exp(-margin)` and then `-exp(-margin)/(1+exp(-margin))`; overflow can produce `-inf/inf = NaN`.

The modern objective instead evaluates `np.logaddexp(0, -margin)`, which does not explicitly exponentiate a large positive number. The gradient uses

\[
\sigma(-m)=\exp[-\log(1+e^{m})]
\]

implemented as `np.exp(-np.logaddexp(0, margin))`. This remains finite even for margins of magnitude millions. Validation rejects non-finite arrays and weights; floating-point overflow in the parameter update raises an exception. Extremely large finite inputs may overflow a dot product before stable activation arithmetic; the corrected implementation detects such overflow and raises `FloatingPointError` with a rescaling diagnostic, rather than silently emitting non-finite results.

## 4. Convexity and conditions on convergence

For the unregularized objective,

\[
\nabla^2F(w)=\frac1n X^\top D X,
\quad D_{ii}=\sigma(m_i)\sigma(-m_i)\in[0,1/4].
\]

The Hessian is positive semidefinite, so \(F\) is convex. With L2 penalty, the Hessian additionally contains \(\lambda P\). A smoothness bound is \(L\leq\|X\|_2^2/(4n)+\lambda\). Standard full-batch GD descent/convergence guarantees require a sufficiently small step, such as \(0<\eta\leq 1/L\), and standard assumptions including a minimizer for claims about convergence to an attained optimum.

**Crucial qualification:** If the unregularized training data are linearly separable, the logistic objective may have infimum zero but **no finite minimizer**; weights can grow indefinitely even as loss approaches zero. Consequently, convexity and a flattening loss plot do **not** establish that the 2024 experiment reached a global minimizer. There is also no basis for claiming an achieved minimum is unique without stronger convexity conditions.

## 5. Decision ties

The historical helper returned `np.sign(X @ w)`, which outputs **zero** for a tie; zero is neither of its valid labels. Historical protocol evaluates `np.sign` exactly, treating zero predictions as unclassified errors and recording a third confusion column if such ties occur. The improved official-split protocol instead assigns +1 at exact ties. Both rules are covered by tests and leave the logistic objective unchanged.
