In the [previous post](/posts/#counting-sat-with-shared-summaries), I chose substitutions that made shared summaries appear. That leaves an important question: how would an algorithm construct them? The paper gives a systematic procedure. Here, I work through its small example.

The example is from **Remark 4.4 of OpenAI's ETH preprint**, using the October 25, 2026 revision. It shows how a matrix construction produces a shared measurement, and how a fixed rule turns that measurement into a polynomial replacement. The general construction appears in Section 4.

## Four coordinates and two evaluation points

I will call the example's four auxiliary coordinates $a,b,c,d$. They correspond to the paper's $\zeta_0,\zeta_1,\zeta_2,\zeta_{12}$. These are coordinates in the construction, rather than the four variables of the formula in the previous post. The full algorithm later places original variables into auxiliary coordinates; repeated copies retain the same value.

We want a polynomial replacement $L(t)$ for the input $a$. It should restore $a$ at $t=0$. At the other two evaluation points, the replacement's assigned group has different coordinates available locally:

| Point | Coordinates available locally |
|---|---|
| $t=1$ | $b,d$ |
| $t=2$ | $c,d$ |

The replacement may also use shared measurements. The task is to make its remaining dependence on other groups pass through those measurements.

## Start with two fixed swaps

Write the coordinate vector as $x=(a,b,c,d)^{\mathsf T}$. The paper's two swap operations act as

$$
\Pi_1x=(b,a,d,c)^{\mathsf T},
$$

$$
\Pi_2x=(c,d,a,b)^{\mathsf T}.
$$

Applying either swap twice restores the original vector. The swaps also commute. For this small example, the construction gives

$$
M=\Pi_1+\Pi_2-\Pi_1\Pi_2.
$$

There is a recipe behind this choice. The example uses $Q=2$ evaluation points and a parameter $R=2$, allowing an auxiliary polynomial of degree at most $R-1=1$. The linear polynomial with $f(0)=1$ and $f(1)=-1$ is $f(z)=1-2z$. Substituting it into the paper's prescription

<div style="overflow-x: auto;"><div style="min-width: max-content;">

$$
W=I-\frac{\Pi_1+\Pi_2}{2},
\qquad V=\Pi_1\Pi_2,
$$

</div></div>

$$
M=Vf(W)
$$

gives the displayed expression for $M$. Here $I$ is the identity operation and $z$ is just the input of the auxiliary polynomial $f$. Evaluating that polynomial on a matrix means $f(W)=I-2W$; expanding it and using $\Pi_1^2=\Pi_2^2=I$ gives $M=\Pi_1+\Pi_2-\Pi_1\Pi_2$.

## Read the summary from the correction

Applying $M$ and subtracting its result from the original vector gives

$$
x-Mx=
\begin{pmatrix}
a-b-c+d\\
-a+b+c-d\\
-a+b+c-d\\
a-b-c+d
\end{pmatrix}.
$$

All four entries are determined by one number:

$$
s=a-b-c+d.
$$

In matrix notation, with $C=(1,-1,-1,1)$, we have $s=Cx$. The correction is simply $(s,-s,-s,s)^{\mathsf T}$. This is the shared measurement: the matrix calculation exposes the same linear quantity in every correction term.

The first coordinate gives the identity we need:

$$
a=s+b+c-d.
$$

So we have written the original input as a shared correction plus three coordinate terms. Next, we remove each coordinate term at any evaluation point where that coordinate is not available locally.

## Cancel a term exactly where it crosses groups

The rule is simple: **to make a term vanish at $t=j$ while preserving it at zero, multiply it by $1-t/j$.**

For our two evaluation points:

| Term | Where it must vanish | Multiplier |
|---|---|---|
| $b$ | $t=2$ | $1-t/2$ |
| $c$ | $t=1$ | $1-t$ |
| $-d$ | Neither point | $1$ |

Apply these multipliers to $a=s+b+c-d$:

<div style="overflow-x: auto;"><div style="min-width: max-content;">

$$
L(t)=s+\left(1-\frac t2\right)b+(1-t)c-d.
$$

</div></div>

Substituting $s=a-b-c+d$ and simplifying gives

$$
\boxed{L(t)=a-\frac t2b-tc.}
$$

At zero, both multipliers equal one, so $L(0)=a$. At the other points,

$$
L(1)=s+\frac12b-d,
$$

$$
L(2)=s-c-d.
$$

Once $s$ is fixed, the first expression uses only the locally available $b,d$, and the second uses only $c,d$. The unwanted dependencies disappear by construction.

For a term that must vanish at several points, the paper multiplies the corresponding factors:

$$
\prod_{j\in E}\left(1-\frac tj\right).
$$

Here $E$ is the set of points where that term crosses out of the assigned group. The degree is $|E|$. An important job of the matrix construction is to keep these sets small.

## Split the measurement before counting

There is one more step. **The quantity $s$ spans both groups.** Directly imposing $s=a-b-c+d$ as one global constraint would still couple their assignments.

At $t=1$, the groups are $\{a,c\}$ and $\{b,d\}$. Split the measurement into their separate contributions:

<div style="overflow-x: auto;"><div style="min-width: max-content;">

$$
q=a-c,\qquad r=d-b,\qquad s=q+r.
$$

</div></div>

We can enumerate the possible pair $(q,r)$, sum assignments within each group having its prescribed value, and combine the group sums. Fixing $q$ constrains only $\{a,c\}$; fixing $r$ constrains only $\{b,d\}$.

For this particular replacement, the expression simplifies further:

$$
L(1)=(a-c)-\frac b2=q-\frac b2.
$$

The remote group communicates through $q$. Its two Boolean assignments $(a,c)=(0,0)$ and $(1,1)$ both give $q=0$, so both contributions must be retained. Sharing a measurement never means dropping multiplicity.

At $t=2$, the groups are $\{a,b\}$ and $\{c,d\}$. The corresponding split is

<div style="overflow-x: auto;"><div style="min-width: max-content;">

$$
q'=a-b,\qquad r'=d-c,\qquad s=q'+r',
$$

</div></div>

and

$$
L(2)=(a-b)-2c=q'-2c.
$$

This group-by-group restriction of the measurement is part of the paper's construction. Other factors may need more of the measurement data than this single replacement does.

## What this example establishes

The swaps produce a shared correction. The vanishing multipliers produce a replacement that preserves the original input at zero and has controlled dependencies at the other points. Those are explicit algebraic operations.

The [verification script](verify_construction.py) constructs the matrices using rational arithmetic, derives the replacement from the cancellation rule, and checks all 16 Boolean assignments. It also checks that assignments with the same local coordinates and remote measurement give the same replacement value.

The general algorithm has to do this for every relevant occurrence, while controlling group sizes, measurement ranges, and polynomial degrees. Section 4 and Lemma 4.7 supply a deterministic construction under the stated parameter conditions. Section 3 then uses grouped sums and Taylor data to recover the original count.

There are two limits to keep in view. First, Remark 4.4 deliberately takes $Q=R=2$, outside the $R<Q$ regime used for the runtime bound. It illustrates the construction without establishing a speedup. Second, this is one replacement row on four auxiliary coordinates, not a complete implementation or independent verification of the paper's counting algorithm.

## Sources

- OpenAI, [*The Exponential Time Hypothesis Is False*][eth], October 25, 2026 revision. Section 4.2 and equation (4.7) give the matrix construction; Section 4.3 and equations (4.9)–(4.12) give the replacements and group restrictions; Remark 4.4 on page 30 gives this example. Lemma 4.7 states the general construction guarantee.
- Ferhat Erata, [*Counting SAT models with shared summaries and interpolation*](/posts/#counting-sat-with-shared-summaries), the preceding worked counting example.

[eth]: https://github.com/openai/math/blob/main/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/paper.pdf
