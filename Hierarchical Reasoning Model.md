---
title: "[]{#_bookmark0 .anchor}Hierarchical Reasoning Model"
---

> Guan Wang^1*,†*^, Jin Li^1^, Yuhao Sun^1^, Xing Chen^1^, Changling
> Liu^1^, Yue Wu^1^, Meng Lu^1*,†*^, Sen Song^2*,†*^, Yasin Abbasi
> Yadkori^1*,†*^
>
> ^1^Sapient Intelligence, Singapore
>
> **Abstract**
>
> Reasoning, the process of devising and executing complex goal-oriented
> action sequences, remains a critical challenge in AI. Current large
> language models (LLMs) primarily employ Chain-of-Thought (CoT)
> techniques, which suffer from brittle task decomposition, extensive
> data requirements, and high latency. Inspired by the hierarchical and
> multi-timescale pro- cessing in the human brain, we propose the
> Hierarchical Reasoning Model (HRM), a novel recurrent architecture
> that attains significant computational depth while maintaining both
> train- ing stability and efficiency. HRM executes sequential reasoning
> tasks in a single forward pass without explicit supervision of the
> intermediate process, through two interdependent recurrent modules: a
> high-level module responsible for slow, abstract planning, and a
> low-level mod- ule handling rapid, detailed computations. With only 27
> million parameters, HRM achieves exceptional performance on complex
> reasoning tasks using only 1000 training samples. The model operates
> without pre-training or CoT data, yet achieves nearly perfect
> performance on challenging tasks including complex Sudoku puzzles and
> optimal path finding in large mazes. Furthermore, HRM outperforms much
> larger models with significantly longer context windows on the
> Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring
> artificial general intelligence capabilities. These results underscore
> HRM's potential as a transformative advancement toward universal
> computation and general-purpose reasoning systems.

![](media/image1.png){width="2.256188757655293in"
height="1.3749496937882764in"}[]{#_bookmark1 .anchor}ARC-AGI-1

> ARC-AGI-2
>
> Sudoku-Extreme (9x9) Maze-Hard (30x30)

960 training examples

> 1120 training examples
>
> 1000 training examples
>
> 1000 training examples
>
> 40 5
>
> 4

30

> 3

20

> 2
>
> 10 1
>
> 0 0 0 0
>
> ![](media/image2.png){width="0.17171369203849518in"
> height="6.666447944006999e-2in"} Chain-of-thought, pretrained
> ![](media/image3.png){width="0.16161307961504812in"
> height="5.656386701662292e-2in"} Direct prediction, small-sample
> learning
>
> Figure 1: **Left:** HRM is inspired by hierarchical processing and
> temporal separation in the brain. It has two recurrent networks
> operating at different timescales to collaboratively solve tasks.
> **Right:** With only about 1000 training examples, the HRM (\~27M
> parameters) surpasses state-of-the-art CoT models on inductive
> benchmarks (ARC-AGI) and challenging symbolic tree-search puzzles
> (*Sudoku-Extreme*, *Maze-Hard*) where CoT models failed completely.
> The HRM was randomly initialized, and it solved the tasks directly
> from inputs without chain of thoughts.
>
> 2Tsinghua University *^†^* Corresponding author. Contact:
> [research@sapient.inc.](mailto:research@sapient.inc) Code available
> at: [github.com/sapientinc/HRM](https://github.com/sapientinc/HRM)

# Introduction

> Deep learning, as its name suggests, emerged from the idea of stacking
> more layers to achieve increased representation power and improved
> performance^1,2^. However, despite the remarkable success of large
> language models, their core architecture is paradoxically shallow^3^.
> This imposes a fundamental constraint on their most sought-after
> capability: reasoning. The fixed depth of stan- dard Transformers
> places them in computational complexity classes such as *AC*^0^ or
> *TC*^0^ ^4^, prevent- ing them from solving problems that require
> polynomial time^5,6^. LLMs are not Turing-complete and thus they
> cannot, at least in a purely end-to-end manner, execute complex
> algorithmic rea- soning that is necessary for deliberate planning or
> symbolic manipulation tasks^7,8^. For example, our results on the
> Sudoku task show that increasing Transformer model depth *can* improve
> per- formance,[^1^](#_bookmark0) but performance remains far from
> optimal even with very deep models (see Figure [2](#_bookmark2)),
> which supports the conjectured limitations of the LLM scaling
> paradigm^9^.
>
> The LLMs literature has relied largely on Chain-of-Thought (CoT)
> prompting for reasoning^10^. CoT externalizes reasoning into
> token-level language by breaking down complex tasks into sim- pler
> intermediate steps, sequentially generating text using a shallow
> model^11^. However, CoT for reasoning is a crutch, not a satisfactory
> solution. It relies on brittle, human-defined decompositions where a
> single misstep or a misorder of the steps can derail the reasoning
> process entirely^12,13^. This dependency on explicit linguistic steps
> tethers reasoning to patterns at the token level. As a result, CoT
> reasoning often requires significant amount of training data and
> generates a large number of tokens for complex reasoning tasks,
> resulting in slow response times. A more efficient approach is needed
> to minimize these data requirements^14^.
>
> Towards this goal, we explore "latent reasoning", where the model
> conducts computations within its internal hidden state space^15,16^.
> This aligns with the understanding that language is a tool for human
> communication, not the substrate of thought itself^17^; the brain
> sustains lengthy, coherent chains of reasoning with remarkable
> efficiency in a latent space, without constant translation back to
> language. However, the power of latent reasoning is still
> fundamentally constrained by a model's *effective computational
> depth*. Naively stacking layers is notoriously difficult due to
> vanishing gra- dients, which plague training stability and
> effectiveness^1,18^. Recurrent architectures, a natural al- ternative
> for sequential tasks, often suffer from early convergence, rendering
> subsequent computa- tional steps inert, and rely on the biologically
> implausible, computationally expensive and memory intensive
> Backpropagation Through Time (BPTT) for training^19^.
>
> The human brain provides a compelling blueprint for achieving the
> effective computational depth that contemporary artificial models
> lack. It organizes computation hierarchically across corti- cal
> regions operating at different timescales, enabling deep, multi-stage
> reasoning^20,21,22^. Recur- rent feedback loops iteratively refine
> internal representations, allowing slow, higher-level areas to guide,
> and fast, lower-level circuits to execute---subordinate processing
> while preserving global coherence^23,24,25^. Notably, the brain
> achieves such depth without incurring the prohibitive credit-
> assignment costs that typically hamper recurrent networks from
> backpropagation through time^19,26^.
>
> Inspired by this hierarchical and multi-timescale biological
> architecture, we propose the Hierar- chical Reasoning Model (HRM). HRM
> is designed to significantly increase the effective compu- tational
> depth. It features two coupled recurrent modules: a high-level (H)
> module for abstract, deliberate reasoning, and a low-level (L) module
> for fast, detailed computations. This structure
>
> 1Simply increasing the model width does not improve performance here.
>
> []{#_bookmark2 .anchor}100 100
>
> 80 80
>
> 60 60
>
> 40 40
>
> 20 20
>
> 27M 54M 109M 218M 436M 872M
>
> Parameters
>
> 8 16 32 64 128 256 512
>
> Depth / Transformer layers computed
>
> Figure 2: **The necessity of depth for complex reasoning. Left:** On
> *Sudoku-Extreme Full*, which require extensive tree-search and
> backtracking, increasing a Transformer's width yields no perfor- mance
> gain, while increasing depth is critical. **Right:** Standard
> architectures saturates, failing to benefit from increased depth. HRM
> overcomes this fundamental limitation, effectively using its
> computational depth to achieve near-perfect accuracy.
>
> avoids the rapid convergence of standard recurrent models through a
> process we term "hierarchi- cal convergence." The slow-updating
> H-module advances only after the fast-updating L-module has completed
> multiple computational steps and reached a local equilibrium, at which
> point the L-module is reset to begin a new computational phase.
>
> Furthermore, we propose a one-step gradient approximation for training
> HRM, which offers im- proved efficiency and eliminates the requirement
> for BPTT. This design maintains a constant mem- ory footprint (*O*(1)
> compared to BPTT's *O*(*T* ) for *T* timesteps) throughout the
> backpropagation process, making it scalable and more biologically
> plausible.
>
> Leveraging its enhanced effective depth, HRM excels at tasks that
> demand extensive search and backtracking. **Using only 1,000
> input-output examples, without pre-training or CoT supervi- sion**,
> HRM learns to solve problems that are intractable for even the most
> advanced LLMs. For example, it achieves near-perfect accuracy in
> complex Sudoku puzzles (*Sudoku-Extreme Full*) and optimal pathfinding
> in 30x30 mazes, where state-of-the-art CoT methods completely fail (0%
> ac- curacy). In the Abstraction and Reasoning Corpus (ARC) AGI
> Challenge^27,28,29^ - a benchmark of inductive reasoning - HRM,
> trained from scratch with only the official dataset (\~1000 exam-
> ples), with only 27M parameters and a 30x30 grid context (900 tokens),
> achieves a performance of **40.3%**, which substantially surpasses
> leading CoT-based models like o3-mini-high (34.5%) and Claude 3.7 8K
> context (21.2%), despite their considerably larger parameter sizes and
> con- text lengths, as shown in Figure [1](#_bookmark1). This
> represents a promising direction toward the development of
> next-generation AI reasoning systems with universal computational
> capabilities.

# Hierarchical Reasoning Model

> We present the HRM, inspired by three fundamental principles of neural
> computation observed in the brain:

- **Hierarchical processing:** The brain processes information across a
  > hierarchy of cortical ar- eas. Higher-level areas integrate
  > information over longer timescales and form abstract repre-
  > sentations, while lower-level areas handle more immediate, detailed
  > sensory and motor process- ing20,22,21.

- **Temporal Separation:** These hierarchical levels in the brain
  > operate at distinct intrinsic timescales, reflected in neural
  > rhythms (e.g., slow theta waves, 4--8 Hz and fast gamma waves,
  > 30--100 Hz)^30,31^. This separation allows for stable, high-level
  > guidance of rapid, low-level computa- tions^32,33^.

- **Recurrent Connectivity:** The brain features extensive recurrent
  > connections. These feedback loops enable iterative refinement,
  > yielding more accurate and context-sensitive representations at the
  > cost of additional processing time. Additionally, the brain largely
  > avoids the problematic deep credit assignment problem associated
  > with BPTT^19^.

> The HRM model consists of four learnable components: an input network
> *f~I~* (*·*; *θ~I~* ), a low-level re- current module *f~L~*(*·*;
> *θ~L~*), a high-level recurrent module *f~H~*(*·*; *θ~H~*), and an
> output network *f~O~*(*·*; *θ~O~*). The model's dynamics unfold over
> *N* high-level cycles of *T* low-level timesteps
> each[^2^](#_bookmark0). We index the total timesteps of one forward
> pass by *i* = 1*, . . . , N × T* . The modules *f~L~* and *f~H~* each
> keep a
>
> hidden state---*z^i^* for *f~L~* and *z^i^* for *f~H~*---which are
> initialized with the vectors *z*^0^ and *z*^0^ , respec-
>
> *L H L H*
>
> tively.
>
> The HRM maps an input vector *x* to an output prediction vector *y*ˆ
> as follows. First, the input *x* is projected into a working
> representation *x*˜ by the input network:
>
> *x*˜ = *f~I~* (*x*; *θ~I~* ) *.*
>
> At each timestep *i*, the L-module updates its state conditioned on
> its own previous state, the H- module's current state (which remains
> fixed throughout the cycle), and the input representation. The
> H-module only updates once per cycle (i.e., every *T* timesteps) using
> the L-module's final state at the end of that cycle:
>
> *z^i^* = *f~L~* *z^i^*^−1^*, z^i^*^−1^*, x*˜; *θ~L~* *,*
>
> (*f~H~* *z^i^*^−1^*, z^i^*^−1^; *θ~H~* if *i ≡* 0 (mod *T* ) *,*
>
> Finally, after *N* full cycles, a prediction *y*ˆ is extracted from
> the hidden state of the H-module:
>
> *y*ˆ = *f~O~*(*z^NT^* ; *θ~O~*) *.*
>
> This entire *NT* -timestep process represents a single forward pass of
> the HRM. A halting mecha- nism (detailed later in this section)
> determines whether the model should terminate, in which case *y*ˆ will
> be used as the final prediction, or continue with an additional
> forward pass.
>
> **Hierarchical convergence** Although convergence is crucial for
> recurrent networks, standard RNNs are fundamentally limited by their
> tendency to converge too early. As the hidden state settles toward a
> fixed point, update magnitudes shrink, effectively stalling subsequent
> computation and capping the network's effective depth. To preserve
> computational power, we actually want convergence to proceed very
> slowly--but engineering that gradual approach is difficult, since
> pushing convergence too far edges the system toward instability.
>
> 2While inspired by temporal separation in the brain, our model's
> "high-level" and "low-level" modules are concep- tual abstractions and
> do not map directly to specific neural oscillation frequencies.

[]{#_bookmark3 .anchor}250

200

150

250

![](media/image4.png){width="1.6829385389326335in"
height="1.0921817585301836in"}200

150

> 250
>
> ![](media/image5.png){width="1.6829385389326335in"
> height="1.0921817585301836in"}![](media/image6.png){width="1.6829385389326335in"
> height="1.0932010061242345in"}200
>
> 150

100

100

> 100
>
> 50 50 50
>
> 0
>
> 0 20 40 60
>
> Step Index \#
>
> ![](media/image7.png){width="1.4733727034120734in"
> height="0.9938462379702537in"}![](media/image8.png){width="0.1002438757655293in"
> height="0.6608158355205599in"}60
>
> 30
>
> 0
>
> 0 20 40 60
>
> ![](media/image9.png){width="1.5065879265091864in"
> height="0.9813560804899387in"}Step Index \#
>
> ![](media/image10.png){width="3.181649168853893e-2in"
> height="0.6559208223972004in"}60
>
> 30

0

200

100

> 0 100 200
>
> Layer Index \#
>
> ![](media/image11.png){width="1.5065879265091864in"
> height="0.9957775590551181in"}![](media/image12.png){width="3.181649168853893e-2in"
> height="0.6583683289588801in"}Principal Components Principal
> Components Principal Components
>
> Figure 3: Comparison of forward residuals and PCA trajectories. HRM
> shows hierarchical conver- gence: the H-module steadily converges,
> while the L-module repeatedly converges within cycles before being
> reset by H, resulting in residual spikes. The recurrent neural network
> exhibits rapid convergence with residuals quickly approaching zero. In
> contrast, the deep neural network experi- ences vanishing gradients,
> with significant residuals primarily in the initial (input) and final
> layers.
>
> HRM is explicitly designed to counteract this premature convergence
> through a process we term *hierarchical convergence*. During each
> cycle, the L-module (an RNN) exhibits stable convergence to a *local
> equilibrium*. This equilibrium, however, depends on the high-level
> state *z~H~* supplied during that cycle. After completing the *T*
> steps, the H-module incorporates the sub-computation's outcome (the
> final state *z~L~*) and performs its own update. This *z~H~* update
> establishes a fresh context for the L-module, essentially "restarting"
> its computational path and initiating a new convergence phase toward a
> different local equilibrium.
>
> This process allows the HRM to perform a sequence of distinct, stable,
> nested computations, where the H-module directs the overall
> problem-solving strategy and the L-module executes the intensive
> search or refinement required for each step. Although a standard RNN
> may approach convergence within *T* iterations, the hierarchical
> convergence benefits from an enhanced effective depth of *NT* steps.
> As empirically shown in Figure [3](#_bookmark3), this mechanism allows
> HRM both to maintain high computational activity (forward residual)
> over many steps (in contrast to a standard RNN, whose activity rapidly
> decays) and to enjoy stable convergence. This translates into better
> performance at any computation depth, as illustrated in Figure
> [2](#_bookmark2).
>
> **Approximate gradient** Recurrent models typically use BPTT to
> compute gradients. However, BPTT requires storing the hidden states
> from the forward pass and then combining them with gradients during
> the backward pass, which demands *O*(*T* ) memory for T timesteps.
> This heavy memory burden forces smaller batch sizes and leads to poor
> GPU utilization, especially for large- scale networks. Additionally,
> because retaining the full history trace through time is biologically
> implausible, it is unlikely that the brain implements BPTT^19^.
>
> Fortunately, if a recurrent neural network converges to a fixed point,
> we can avoid unrolling its state sequence by applying backpropagation
> in a single step at that equilibrium point. Moreover, such a mechanism
> could plausibly be implemented in the brain using only local learning
> rules^34,35^. Based
>
> on this finding, we propose a one-step approximation of the HRM
> gradient--using the gradient of the last state of each module and
> treating other states as constant. The gradient path is, therefore,

Output head → final state of the H-module → final state of the L-module
→ input embedding

> ![](media/image13.png){width="2.549407261592301in"
> height="0.9703291776027997in"}The above method needs *O*(1) memory,
> does not require unrolling through time, and can be easily implemented
> with an autograd framework such as PyTorch, as shown in Figure
> [4](#_bookmark4). Given that each module only needs to back-propagate
> errors through its most recent local synaptic activity, this approach
> aligns well with the perspective that cortical credit assignment
> relies on short-range, temporally local mechanisms rather than on a
> global replay of activity patterns.
>
> The one-step gradient approximation is theoretically grounded in the
> mathematics of Deep Equilibrium Mod- els (DEQ)^36^ which employs the
> Implicit Function Theo- rem (IFT) to bypass BPTT, as detailed next.
> Consider an idealized HRM behavior where, during high-level cycle
>
> *k*, the L-module repeatedly updates until its state *z~L~* con-
> verges to a local fixed point *z^⋆^* . This fixed point, given the
> current high-level state *z^k^*^−1^, can be expressed as
>
> *z^⋆^* = *f* (*z^⋆^ , z^k^*^−1^*, x*˜; *θ* ) *.*
>
> []{#_bookmark4 .anchor}def hrm(z, x, N=2, T=2): x = input_embedding(x)
> zH, zL = z
>
> with torch.no_grad():
>
> for \_i in range(N ∗ T − 1): zL = L_net(zL, zH, x)
>
> *L L L H L*
>
> The H-module then performs a single update using this converged
> L-state:
>
> *z^k^* = *f~H~*(*z^k^*^−1^*, z^⋆^* ; *θ~H~*) *.*

if (\_i + 1) % T == 0:

zH = H_net(zH, zL)

> \# 1−step grad
>
> zL = L_net(zL, zH, x) zH = H_net(zH, zL)
>
> return (zH, zL), output_head(zH)
>
> \# Deep Supervision
>
> *H H L*
>
> for x, y_true in train_dataloader:
>
> With a proper mapping *F*, the updates to the high-level state can be
> written in a more compact form as *z^k^* = *F*(*z^k^*^−1^; *x*˜*, θ*),
> where *θ* = (*θ~I~ , θ~L~*), and the fixed-point can be written as
> *z^⋆^* = *F*(*z^⋆^* ; *x*˜*, θ*). Let *J*~F~ = ^[*∂*F]{.underline}^ be

*H*

> z = z_init
>
> for step in range(N_supervision): z, y_hat = hrm(z, x)
>
> loss = softmax_cross_entropy(y_hat, y_true) z = z.detach()
>
> loss.backward()
>
> the Jacobian of *F*, and assume that the matrix *I − J*~F~ is
>
> invertible at *z^⋆^* and that the mapping *F* is continuously
>
> opt.step() opt.zero_grad()
>
> differentiable. The Implicit Function Theorem then al- lows us to
> calculate the exact gradient of fixed point *z^⋆^* with respect to the
> parameters *θ* without explicit back- propagation:
>
> Figure 4: **Top:** Diagram of HRM with approximate gradient.
> **Bottom:** Pseu- docode of HRM with deep supervision training in
> PyTorch.

[]{#_bookmark5 .anchor}*∂z^⋆^*

*∂θ*

> = (*I − J*~F~ I*z⋆*
>
> −1 *[∂F]{.underline}*
>
> *∂θ ⋆ H*
>
> *.* (1)
>
> Calculating the above gradient requires evaluating and inverting
> matrix (*I − J*~F~ ) that can be com- putationally expensive. Given
> the Neumann series expansion,
>
> (*I − J*~F~ )^−1^ = *I* + *J*~F~ + *J*^2^ + *J*^3^ + *. . . ,*
>
> F F
>
> the so-called *1-step gradient* ^37^ approximates the series by
> considering only its first term, i.e. (*I −*
>
> *J*~F~ )^−1^ *≈ I*, and leads to the following approximation of
> Equation ([1](#_bookmark5)):
>
> []{#_bookmark6 .anchor} *∂z*^∗^
>
> *∂θ~H~*
>
> *≈ ∂f~H~ ,*
>
> *∂θ~H~*
>
> *∂z*^∗^
>
> *∂θ~L~*
>
> *∂f~H~*
>
> *≈ ∂z*^∗^
>
> *∂z*^∗^
>
> *· ^[L]{.underline}^ ,*
>
> *∂θ~L~*
>
> *∂z*^∗^
>
> *∂θ~I~*
>
> *∂f~H~*
>
> *≈ ∂z*^∗^
>
> *∂z*^∗^
>
> *· [L]{.underline}*
>
> *∂θ~I~*
>
> *.* (2)
>
> The gradients of the low-level fixed point, *∂z∗*

*L*

> and *∂z∗* , can also be approximated using another
>
> *I*
>
> application of the 1-step gradient:
>
> []{#_bookmark7 .anchor} *∂z*^∗^
>
> *∂θ~L~*
>
> *≈ ∂f~L~ ,*
>
> *∂θ~L~*
>
> *∂z*^∗^
>
> *∂θ~I~*
>
> *≈ ∂f~L~*
>
> *∂θ~I~*
>
> *.* (3)
>
> By substituting Equation ([3](#_bookmark7)) back into Equation
> ([2](#_bookmark6)), we arrive at the final simplified gradients.
>
> Before defining our loss function, we must first introduce two key
> elements of our proposed method: *deep supervision* and *adaptive
> computational time*.
>
> **Deep supervision** Inspired by the principle that periodic neural
> oscillations regulate when learning occurs in the brain^38^, we
> incorporate a deep supervision mechanism into HRM, as detailed next.
>
> Given a data sample (*x, y*), we run multiple forward passes of the
> HRM model, each of which we refer to as a *segment*. Let *M* denote
> the total number of segments executed before termination. For each
> segment *m ∈ {*1*, . . . , M}*, let *z^m^* = (*z^mNT^ , z^mNT^* )
> represent the hidden state at the
>
> *H L*
>
> conclusion of segment *m*, encompassing both high-level and low-level
> state components. At each segment *m*, we apply a deep supervision
> step as follows:

1.  Given the state *z^m^*^−1^ from the previous segment, compute the
    > next state *z^m^* and its associated

> output *y*ˆ*^m^* through a forward pass in the HRM model:
>
> (*z^m^, y*ˆ*^m^*) *←* HRM(*z^m^*^−1^*, x*; *θ*)

2.  Compute the loss for the current segment:

> *L^m^ ←* LOSS(*y*ˆ*^m^, y*)

3.  Update parameters:

> *θ ←* [OptimizerStep(]{.smallcaps}*θ, ∇~θ~L^m^*[)]{.smallcaps}
>
> The crucial aspect of this procedure is that the hidden state *z^m^*
> is "detached" from the computa- tion graph before being used as the
> input state for the next segment. Consequently, gradients from segment
> *m* + 1 do not propagate back through segment *m*, effectively
> creating a 1-step approxi- mation of the gradient of the recursive
> deep supervision process^39,40^. This approach provides more frequent
> feedback to the H-module and serves as a regularization mechanism,
> demonstrating supe- rior empirical performance and enhanced stability
> in deep equilibrium models when compared to more complex,
> Jacobian-based regularization techniques^39,41^. Figure
> [4](#_bookmark4) shows pseudocode of deep supervision training.
>
> **Adaptive computational time (ACT)** The brain dynamically alternates
> between automatic think- ing ("System 1") and deliberate reasoning
> ("System 2")^42^. Neuroscientific evidence shows that these cognitive
> modes share overlapping neural circuits, particularly within regions
> such as the prefrontal cortex and the default mode network^43,44^.
> This indicates that the brain dynamically mod- ulates the "runtime" of
> these circuits according to task complexity and potential
> rewards^45,46^.
>
> Inspired by the above mechanism, we incorporate an adaptive halting
> strategy into HRM that en- ables "thinking, fast and slow". This
> integration leverages deep supervision and uses the Q-learning
>
> algorithm^47^ to adaptively determine the number of segments. A Q-head
> uses the final state of the
>
> H-module to predict the Q-values *Q*ˆ*m* = (*Q*ˆ*m , Q*ˆ*m* ) of the
> "halt" and "continue" actions:
>
> halt continue
>
> *Q*ˆ*m* = *σ*(*θ*^⊤^*z^mNT^* ) *,*
>
> *Q H*
>
> where *σ* denotes the sigmoid function applied element-wise. The halt
> or continue action is chosen using a randomized strategy as detailed
> next. Let *M*~max~ denote the maximum number of segments (a fixed
> hyperparameter) and *M*~min~ denote the minimum number of segments (a
> random variable). The value of *M*~min~ is determined stochastically:
> with probability *ε*, it is sampled uniformly from the set *{*2*, · ·
> · , M*~max~*}* (to encourage longer thinking), and with probability 1
> *−ε*, it is set to 1. The halt action is selected under two
> conditions: when the segment count surpasses the maximum threshold
> *M*~max~, or when the estimated halt value *Q*ˆhalt exceeds the
> estimated continue value *Q*ˆcontinue and the segment count has
> reached at least the minimum threshold *M*~min~.
>
> The Q-head is updated through a Q-learning algorithm, which is defined
> on the following episodic Markov Decision Process (MDP). The state of
> the MDP at segment *m* is *z^m^*, and the action space is *{*halt*,*
> continue*}*. Choosing the action "halt" terminates the episode and
> returns a binary reward indicating prediction correctness, i.e.,
> **1***{y*ˆ*^m^* = *y}*. Choosing "continue" yields a reward of 0 and
>
> the state transitions to *z^m^*^+1^. Thus, the Q-learning targets for
> the two actions *G*ˆ*m* = (*G*ˆ*m , G*ˆ*m* )
>
> are given by
>
> ˆ*m*

halt

> = **1***{y*ˆ*^m^* = *y} ,*
>
> *Q ,* if *m ≥ N*~max~ *,*

halt

> continue
>
> max(*Q*ˆhalt *, Q*ˆcontinue) *,* otherwise *.*
>
> We can now define the loss function of our learning procedure. The
> overall loss for each supervision segment combines both the Q-head
> loss and the sequence-to-sequence loss:
>
> *m*
>
> ACT
>
> [= Loss(]{.smallcaps}*y*[ˆ]{.smallcaps}*^m^, y*[) +
> BinaryCrossEntropy(]{.smallcaps}*Q*[ˆ]{.smallcaps}*m,
> G*[ˆ]{.smallcaps}*m*[)]{.smallcaps} *.*
>
> Minimizing the above loss enables both accurate predictions and nearly
> optimal stopping decisions.
>
> Selecting the "halt" action ends the supervision loop. In practice,
> sequences are processed in batches, which can be easily handled by
> substituting any halted sample in the batch with a fresh sample from
> the dataloader.
>
> Figure [5](#_bookmark8) presents a performance comparison between two
> HRM variants: one incorporating ACT and another employing a fixed
> computational step count equivalent to ACT's *M*~max~ parameter. It
> shows that ACT effectively adapts its computational resources based on
> task complexity, achieving significant computational savings with
> minimal impact on performance.
>
> **Inference-time scaling** An effective neural model should exploit
> additional computational re- sources during inference to enhance
> performance. As illustrated in Figure [5](#_bookmark8)-(c), HRM
> seamlessly achieves inference-time scaling by simply increasing the
> computational limit parameter, *M*~max~ without requiring further
> training or architectural modifications.
>
> Additional compute is especially effective for tasks that demand
> deeper reasoning. On Sudoku--- a problem that often requires long-term
> planning---HRM exhibits strong inference-time scaling. On the other
> hand, we find that extra computational resources yield minimal gains
> in ARC-AGI challenge, as solutions generally require only a few
> transformations.

a.  []{#_bookmark8 .anchor}ACT Compute Spent

b.  ACT Performance

c.  Inference-time scaling

8

7

6

100.0

97.5

95.0

> 100.0
>
> ![](media/image14.png){width="1.7802220034995626in"
> height="0.9723020559930009in"}97.5
>
> 95.0

5

92.5

> 92.5
>
> 4
>
> 3
>
> 2
>
> 1
>
> 2 4 8
>
> *M* (Fixed) or *M*max (ACT)

90.0

87.5

85.0

82.5

> 2 4 8
>
> *M* (Fixed) or *M*max (ACT)

90.0

87.5

85.0

82.5

> 2 4 8 16
>
> Inference *M*max
>
> Figure 5: **Effectiveness of Adaptive Computation Time (ACT)** on the
> *Sudoku-Extreme-Full*. **(a)** Mean compute steps used by models with
> ACT versus models with a fixed number of compute steps (*M* ). ACT
> maintains a low and stable number of average compute steps even as the
> maximum limit (*M*~max~) increases. **(b)** Accuracy comparison. The
> ACT model achieves performance comparable to the fixed-compute model
> while utilizing substantially fewer computational steps on average.
> **(c)** Inference-time scalability. Models trained with a specific
> *M*~max~ can generalize to higher compu- tational limits during
> inference, leading to improved accuracy. For example, a model trained
> with *M*~max~ = 8 continues to see accuracy gains when run with
> *M*~max~ = 16 during inference.
>
> **Stability of Q-learning in ACT** The deep Q-learning that underpins
> our ACT mechanism is known to be prone to instability, often requiring
> stabilization techniques such as replay buffers and target
> networks^48^, which are absent in our design. Our approach, however,
> achieves stability through the intrinsic properties of our model and
> training procedure. Recent theoretical work by Gallici et al. ^49^
> shows that Q-learning can achieve convergence if network parameters
> are bounded, weight decay is incorporated during training, and
> post-normalization layers are implemented. Our model satisfies these
> conditions through its Post-Norm architecture that employs RMSNorm (a
> layer normalization variant) and the AdamW optimizer. AdamW has been
> shown to solve an *L*~∞~- constrained optimization problem, ensuring
> that model parameters remain bounded by 1*/λ*^50^.
>
> **Architectural details** We employ a sequence-to-sequence
> architecture for HRM. Both input and output are represented as token
> sequences: *x* = (*x*~1~*, . . . , x~l~*) and *y* = (*y*~1~*, . . . ,
> y~l~′* ) respectively. The model includes an embedding layer *f~I~*
> that converts discrete tokens into vector representa- tions, and an
> output head *f~O~*(*z*; *θ~O~*) = softmax(*θ~O~z*) that transforms
> hidden states into token prob- ability distributions *y*ˆ. For
> small-sample experiments, we replace softmax with stablemax^51^ to
> improve generalization performance. The sequence-to-sequence loss is
> averaged over all tokens,
>
> LOSS(*y*ˆ*, y*) = ^[1]{.underline}^ I:*l′* log *p*(*y* ), where
> *p*(*y* ) is the probability that distribution *y*ˆ assigns to token
>
> *y~i~*. The initial hidden states *z*^0^ are initialized by sampling
> from a truncated normal distribution with standard deviation of 1,
> truncation of 2, and kept fixed throughout training.
>
> Both the low-level and high-level recurrent modules *f~L~* and *f~H~*
> are implemented using encoder- only Transformer^52^ blocks with
> identical architectures and dimensions. These modules take mul- tiple
> inputs, and we use straightforward element-wise addition to combine
> them, though more sophisticated merging techniques such as gating
> mechanisms could potentially improve perfor- mance and is left for
> future work. For all Transformer blocks in this work---including those
> in the baseline models---we incorporate the enhancements found in
> modern LLMs (based on Llama^53^ architectures). These improvements
> include Rotary Positional Encoding^54^, Gated Linear Units^55^,
> RMSNorm^56^, and the removal of bias terms from linear layers.
>
> Furthermore, both HRM and recurrent Transformer models implement a
> Post-Norm architecture

![](media/image15.png){width="0.6628958880139982in"
height="0.6621380139982502in"}![](media/image16.png){width="0.6644116360454944in"
height="0.6621380139982502in"}

> ![](media/image17.png){width="0.6628958880139982in"
> height="0.6621391076115486in"}![](media/image18.png){width="0.6644116360454944in"
> height="0.6621391076115486in"}![](media/image19.png){width="0.6628958880139982in"
> height="0.6636537620297462in"}![](media/image20.png){width="0.6644116360454944in"
> height="0.6575929571303587in"}![](media/image21.png){width="1.1137489063867017in"
> height="1.0040616797900261in"}![](media/image22.png){width="1.1128991688538932in"
> height="1.0040616797900261in"}![](media/image23.jpeg){width="1.7024234470691164in"
> height="2.2083333333333335in"}[]{#_bookmark9 .anchor}(a) ARC-AGI (b)
> Sudoku-Hard (c) Maze navigation (d) *Sudoku-Extreme* subset difficulty
>
> Figure 6: **Left:** Visualization of benchmark tasks. **Right:**
> Difficulty of *Sudoku-Extreme* examples.
>
> with weights initialized via truncated LeCun Normal
> initialization^57,58,59^, while the scale and bias parameters are
> excluded from RMSNorm. All parameters are optimized using the
> Adam-atan2 op- timizer^60^, a scale-invariant variant of Adam^61^,
> combined with a constant learning rate that includes linear warm-up.

# Results

> This section begins by describing the ARC-AGI, Sudoku, and Maze
> benchmarks, followed by an overview of the baseline models and their
> results. Figure [6](#_bookmark9)-(a,b,c) presents a visual representa-
> tion of the three benchmark tasks, which are selected to evaluate
> various reasoning abilities in AI models.

## Benchmarks

> **ARC-AGI Challenge** The ARC-AGI benchmark evaluates general fluid
> intelligence through IQ- test-like puzzles that require inductive
> reasoning^27^. The initial version, ARC-AGI-1, presents chal- lenges
> as input-label grid pairs that force AI systems to extract and
> generalize abstract rules from just a few examples. Each task provides
> a few input--output demonstration pairs (usually 2--3) and a test
> input. An AI model has two attempts to produce the correct output
> grid. Although some be- lieve that mastering ARC-AGI would signal true
> artificial general intelligence, its primary purpose is to expose the
> current roadblocks in AGI progress. In fact, both conventional deep
> learning meth- ods and CoT techniques have faced significant
> challenges with ARC-AGI-1, primarily because it requires the ability
> to generalize to entirely new tasks^28^.
>
> Addressing the limitations identified in ARC-AGI-1, ARC-AGI-2
> significantly expands the bench- mark by providing a more
> comprehensive and carefully refined collection of tasks. These new
> tasks emphasize deeper compositional reasoning, multi-step logic,
> contextual rule application, and symbolic abstraction. Human
> calibration studies show these tasks are challenging but doable for
> people, while being much harder for current AI systems, offering a
> clearer measure of general reasoning abilities^29^.
>
> **Sudoku-Extreme** Sudoku is a 9*×*9 logic puzzle, requiring each row,
> column, and 3*×*3 block to contain the digits 1--9 exactly once. A
> prediction is considered correct if it exactly matches the puzzle's
> unique solution. Sudoku's complex logical structure makes it a popular
> benchmark for evaluating logical reasoning in machine
> learning^62,63,64^.
>
> The most frequently used Sudoku dataset in research, namely the Kaggle
> dataset^65^, can be fully solved using elementary single-digit
> techniques^66^. The minimal 17-clue puzzles^62^, another widely- used
> collection, might seem more challenging due to its small number of
> clues. However, this perception is misleading---since 17 represents
> the minimum number of clues required to guarantee a unique Sudoku
> solution, these hints need to be highly orthogonal to each other. This
> orthogonal arrangement leads to many direct, easily-resolved solution
> paths^67^.
>
> We introduce *Sudoku-Extreme*, a more challenging dataset that is
> compiled from the aforemen- tioned easy datasets as well as puzzles
> recognized by the Sudoku community as exceptionally difficult for
> human players:

- Easy puzzles compiled from Kaggle, 17-clue, plus unbiased samples from
  > the Sudoku puzzle distribution^67^: totaling 1 149 158 puzzles.

- Challenging puzzles compiled from Magictour 1465, Forum-Hard and
  > Forum-Extreme subsets: totaling 3 104 157 puzzles.

> The compiled data then undergo a strict 90/10 train-test split,
> ensuring that the test set puzzles cannot be derived through
> equivalent transformations of any training samples. *Sudoku-Extreme*
> is a down-sampled subset of this data containing 1000 training
> examples. We use *Sudoku-Extreme* in our main experiments (Figure
> [1](#_bookmark1)), which focuses on small-sample learning scenarios.
> To guarantee convergence and control overfitting effects in our
> analysis experiments (Figures [2](#_bookmark2), [3](#_bookmark3) and
> [5](#_bookmark8)), we use the complete training data,
> *Sudoku-Extreme-Full*, containing 3 831 994 examples.
>
> We measure puzzle difficulty by counting the number of search
> backtracks ("guesses") required by a smart Sudoku solver program
> *tdoku*, which uses propositional logic to reduce the number of
> guesses^67^. Our *Sudoku-Extreme* dataset exhibits a mean difficulty
> of 22 backtracks per puzzle, sig- nificantly higher than existing
> datasets, including recent handmade puzzles Sudoku-Bench^68^ which
> average just 0*.*45 backtracks per puzzle. These subset complexity
> levels are shown in Figure [6](#_bookmark9)-(d).
>
> **Maze-Hard** This task involves finding the optimal path in a 30*×*30
> maze, making it interpretable and frequently used for training LLMs in
> search tasks^69,70,71^. We adopt the instance generation procedure of
> Lehnert et al. ^71^, but introduce an additional filter to retain only
> those instances whose difficulty exceeds 110. Here, "difficulty" is
> defined as the length of the shortest path, which aligns with the
> linear time complexity of the wavefront breadth-first search algorithm
> on GPUs^72^. A path is considered correct if it is valid and
> optimal---that is, the shortest route from the start to the goal. The
> training and test set both include 1000 examples.

## Evaluation Details

> For all benchmarks, HRM models were initialized with random weights
> and trained in the sequence- to-sequence setup using the input-output
> pairs. The two-dimensional input and output grids were flattened and
> then padded to the maximum sequence length. The resulting performance
> is shown in Figure [1](#_bookmark1). Remarkably, HRM attains these
> results with just \~1000 training examples per task---and **without
> pretraining or CoT labels**.
>
> For ARC-AGI challenge, we start with (1) all demonstration and test
> input-label pairs from the training set, and (2) all demonstration
> pairs along with test inputs from the evaluation set. The dataset is
> augmented by applying translations, rotations, flips, and color
> permutations to the puz- zles. Each task example is prepended with a
> learnable special token that represents the puzzle it belongs to. At
> test time, we proceed as follows for each test input in the evaluation
> set: (1) Gener- ate and solve 1000 augmented variants and, for each,
> apply the inverse-augmentation transform to obtain a prediction. (2)
> Choose the two most popular predictions as the final
> outputs.[^3^](#_bookmark0) All reported results are obtained by
> comparing the outputs with the withheld test labels from the
> evaluation set.
>
> We augment Sudoku puzzles by applying band and digit permutations,
> while data augmentation is disabled for Maze tasks. Both tasks undergo
> only a single inference pass.
>
> For ARC-AGI, the scores of the CoT models are taken from the official
> leaderboard^29^, while for Sudoku and Maze, the scores are obtained by
> evaluating through the corresponding API.
>
> In Figure [1](#_bookmark1), the baselines are grouped based on whether
> they are pre-trained and use CoT, or neither. The "Direct pred"
> baseline means using "direct prediction without CoT and pre-training",
> which retains the exact training setup of HRM but swaps in a
> Transformer architecture. Interestingly, on ARC-AGI-1, "Direct pred"
> matches the performance of Liao and Gu ^73^, who built a carefully de-
> signed, domain-specific equivariant network for learning the ARC-AGI
> task from scratch, without pre-training. By substituting the
> Transformer architecture with HRM's hierarchical framework and
> implementing ACT, we achieve more than a twofold performance
> improvement.
>
> On the *Sudoku-Extreme* and *Maze-Hard* benchmarks, the performance
> gap between HRM and the baseline methods is significant, as the
> baselines almost never manage to solve the tasks. These benchmarks
> that demand lengthy reasoning traces are particularly difficult for
> CoT-based methods. With only 1000 training examples, the "Direct pred"
> baseline---which employs an 8-layer Trans- former identical in size to
> HRM---fails entirely on these challenging reasoning problems. When
> trained on the larger *Sudoku-Extreme-Full* dataset, however, "Direct
> pred" can solve some easy Sudoku puzzles and reaches 16*.*9% accuracy
> (see Figure [2](#_bookmark2)). Lehnert et al. ^71^ showed that a large
> vanilla Transformer model with 175M parameters, trained on 1 million
> examples across multiple trials, achieved only marginal success on
> 30x30 Maze tasks, with accuracy below 20% using the *pass*@64
> evaluation metric.

## Visualization of intermediate timesteps

> Although HRM demonstrates strong performance on complex reasoning
> tasks, it raises an intrigu- ing question: what underlying reasoning
> algorithms does the HRM neural network actually imple- ment?
> Addressing this question is important for enhancing model
> interpretability and developing a deeper understanding of the HRM
> solution space.
>
> While a definitive answer lies beyond our current scope, we begin our
> investigation by analyzing state trajectories and their corresponding
> solution evolution. More specifically, at each timestep
>
> *i* and given the low-level and high-level state pair (*z^i^* and
> *z^i^* ) we perform a preliminary forward
>
> pass through the H-module to obtain *z*¯*^i^* = *f~H~*(*z^i^ , z^i^* ;
> *θ~H~*) and its corresponding decoded prediction
>
> *H L*
>
> *y*¯*^i^* = *f~O~*(*z*¯*^i^*; *θ~O~*). The prediction *y*¯*^i^* is
> then visualized in Figure [7](#_bookmark10).
>
> In the Maze task, HRM appears to initially explore several potential
> paths simultaneously, subse- quently eliminating blocked or
> inefficient routes, then constructing a preliminary solution outline
>
> 3The ARC-AGI allows two attempts for each test input.
>
> ![](media/image24.png){width="0.9739577865266842in"
> height="0.9739577865266842in"}![](media/image24.png){width="0.9739577865266842in"
> height="0.9739577865266842in"}![](media/image25.png){width="0.9739577865266842in"
> height="0.9739577865266842in"}![](media/image26.png){width="0.9739577865266842in"
> height="0.9739577865266842in"}![](media/image27.png){width="0.9739577865266842in"
> height="0.9739577865266842in"}![](media/image28.png){width="0.9739577865266842in"
> height="0.9739577865266842in"}![](media/image29.png){width="0.9739577865266842in"
> height="0.9739577865266842in"}T[]{#_bookmark10 .anchor}imestep *i* = 0
> Timestep *i* = 1 Timestep *i* = 2 Timestep *i* = 3 Timestep *i* = 4
> Timestep *i* = 5 Timestep *i* = 6

Initial

Timestep *i* = 0

> Timestep *i* = 1
>
> Timestep *i* = 2
>
> Timestep *i* = 3
>
> Timestep *i* = 4
>
> Timestep *i* = 5
>
> Timestep *i* = 6
>
> Timestep *i* = 7
>
> ![](media/image30.png){width="0.7621872265966754in"
> height="0.7621872265966754in"}![](media/image31.png){width="0.7621872265966754in"
> height="0.7621872265966754in"}![](media/image32.png){width="0.7621872265966754in"
> height="0.7621872265966754in"}![](media/image33.png){width="0.7621872265966754in"
> height="0.7621872265966754in"}![](media/image34.png){width="0.7621872265966754in"
> height="0.7621872265966754in"}![](media/image35.png){width="0.7621872265966754in"
> height="0.7621872265966754in"}![](media/image36.png){width="0.7621872265966754in"
> height="0.7621872265966754in"}![](media/image37.png){width="0.7621872265966754in"
> height="0.7621872265966754in"}\[7666fa5d\] Example Input \[7666fa5d\]
> Example Output \[7666fa5d\] Test Input Timestep *i* = 0 Timestep *i* =
> 1 Timestep *i* = 2 Timestep *i* = 3 Timestep *i* = 4
>
> ![](media/image38.png){width="0.3590638670166229in"
> height="0.6125207786526684in"}![](media/image39.png){width="0.3590638670166229in"
> height="0.6125207786526684in"}![](media/image40.png){width="0.3590638670166229in"
> height="0.6125207786526684in"}![](media/image41.png){width="0.3590638670166229in"
> height="0.6125207786526684in"}![](media/image42.png){width="0.3590638670166229in"
> height="0.6125207786526684in"}![](media/image43.png){width="0.3590638670166229in"
> height="0.6125207786526684in"}![](media/image43.png){width="0.3590638670166229in"
> height="0.6125207786526684in"}![](media/image44.png){width="0.3590638670166229in"
> height="0.6125207786526684in"}\[7b80bb43\] Test Input Timestep *i* = 0
> Timestep *i* = 1 Timestep *i* = 2 Timestep *i* = 3 Timestep *i* = 4
> Timestep *i* = 5 Timestep *i* = 6
>
> ![](media/image45.png){width="1.441000656167979in"
> height="0.375in"}\[7b80bb43\] Example Input \[7b80bb43\] Example
> Output
>
> Figure 7: **Visualization of intermediate predictions by HRM on
> benchmark tasks. Top:** *Maze- Hard*---blue cells indicate the
> predicted path. **Middle:** *Sudoku-Extreme*---bold cells represent
> ini- tial givens; red highlights cells violating Sudoku constraints;
> grey shading indicates changes from the previous timestep. **Bottom:**
> ARC-AGI-2 Task---left: provided example input-output pair; right:
> intermediate steps solving the test input.
>
> followed by multiple refinement iterations. In Sudoku, the strategy
> resembles a depth-first search approach, where the model appears to
> explore potential solutions and backtracks when it hits dead ends. HRM
> uses a different approach for ARC tasks, making incremental
> adjustments to the board and iteratively improving it until reaching a
> solution. Unlike Sudoku, which involves frequent backtracking, the ARC
> solution path follows a more consistent progression similar to
> hill-climbing optimization.
>
> Importantly, the model shows that it can adapt to different reasoning
> approaches, likely choosing an effective strategy for each particular
> task. Further research is needed to gain more comprehensive insights
> into these solution strategies.

# Brain Correspondence

> A key principle from systems neuroscience is that a brain region's
> functional repertoire---its ability to handle diverse and complex
> tasks---is closely linked to the dimensionality of its neural
> represen- tations^75,76^. Higher-order cortical areas, responsible for
> complex reasoning and decision-making, must handle a wide variety of
> tasks, demanding more flexible and context-dependent processing^77^.
> In dynamical systems, this flexibility is often realized through
> higher-dimensional state-space tra- jectories, which allow for a
> richer repertoire of potential computations^78^. This principle gives
> rise to an observable *dimensionality hierarchy*, where a region's
> position in the processing hierarchy

[]{#_bookmark11 .anchor}(a)

\(b\)

> ![](media/image46.jpeg){width="1.675123578302712in"
> height="1.205728346456693in"}
>
> ![](media/image47.jpeg){width="0.9677121609798776in"
> height="1.349857830271216in"}![](media/image48.png){width="0.14688538932633421in"
> height="1.1016108923884513in"}5.0
>
> 4.5
>
> 4.0
>
> 3.5
>
> 3.0
>
> 2.5
>
> 2.0
>
> \(c\)

![](media/image49.jpeg){width="1.9418678915135608in"
height="1.1293744531933507in"}

> \(d\)
>
> \(e\)

![](media/image50.jpeg){width="1.947567804024497in"
height="1.1293744531933507in"}

> \(f\)

![](media/image51.png){width="1.996809930008749in"
height="1.1716502624671916in"}![](media/image52.png){width="1.9974650043744533in"
height="1.171185476815398in"}0 20 40

Position in the hierarchy

> Figure 8: **Hierarchical Dimensionality Organization in the HRM and
> Mouse Cortex.** (a,b) are adapted from Posani et al. ^74^. (a)
> Anatomical illustration of mouse cortical areas, color-coded by
> functional modules. (b) Correlation between Participation Ratio (PR),
> a measure of effective neural dimensionality, and hierarchical
> position across different mouse cortical areas. Higher positions in
> the hierarchy (e.g., MOs, ACAd) exhibit significantly higher PR values
> compared to lower sensory areas (e.g., SSp-n), with a Spearman
> correlation coefficient of *ρ* = 0.79 (P = 0.0003). (c,d) **Trained
> HRM.** (c) PR scaling of the trained HRM with task diversity. The
> dimensionality of the high- level module (*z~H~*) scales with the
> number of unique tasks (trajectories) included in the analysis,
> indicating an adaptive expansion of its representational capacity. In
> contrast, the low-level module's (*z~L~*) dimensionality remains
> stable. (d) PR values for the low-level (*z~L~*, PR = 30.22) and high-
> level (*z~H~*, PR = 89.95) modules of the *trained* HRM, computed from
> neural activity during 100 unique Sudoku-solving trajectories. A clear
> dimensionality hierarchy is observed, with the high- level module
> operating in a substantially higher-dimensional space. (e,f)
> **Analysis of Untrained Network.** To verify that the dimensionality
> hierarchy is an emergent property of training, the same analyses were
> performed on an *untrained* HRM with random weights. (e) In contrast
> to the trained model's scaling in (c), the dimensionality of both
> modules in the untrained model remains low and stable, failing to
> scale with the number of tasks. (f) Similarly, contrasting with the
> clear separation in (d), the PR values for the untrained model's
> modules (*z~L~*, PR = 42.09; *z~H~*, PR = 40.75) are low and nearly
> identical, showing no evidence of hierarchical separation. This
> confirms that the observed hierarchical organization of dimensionality
> is a learned property that emerges through training, not an artifact
> of the model's architecture.
>
> correlates with its *effective dimensionality*. To quantify this
> phenomenon, we can examine the Participation Ratio (PR), which serves
> as a standard measure of the effective dimensionality of a
> high-dimensional representation^79^. The PR is calculated using the
> formula

PR (I:*i λ~i~*)^2^

> where *{λ~i~}* are the eigenvalues of the covariance matrix of neural
> trajectories. Intuitively, a higher PR value signifies that variance
> is distributed more evenly across many dimensions, corresponding to a
> higher-dimensional representation. Conversely, a lower PR value
> indicates that variance is concentrated in only a few principal
> components, reflecting a more compact, lower-dimensional structure.
>
> The dimensionality hierarchy can be observed, for example, in the
> mouse cortex, where the PR of population activity increases
> monotonically from low-level sensory areas to high-level associative
> areas, supporting this link between dimensionality and functional
> complexity^74^ (Figure [8](#_bookmark11) (a,b)).
>
> We evaluated whether HRM reproduces this neuroscientific principle by
> calculating the PR for both recurrent modules after training on the
> *Sudoku-Extreme Full* dataset. The PR computation used the covariance
> matrix derived from neural states gathered across multiple
> Sudoku-solving trajectories. The results show a striking parallel to
> the biological findings. The low-level module's state (*z~L~*)
> occupies a relatively small subspace with a participation ratio of
> 30.22, whereas the high- level module's state (*z~H~*) operates in a
> substantially larger subspace with a participation ratio of 89.95, as
> shown in Figure [8](#_bookmark11)(c). Furthermore, Figure
> [8](#_bookmark11)(d) shows that increasing the number of unique tasks
> (trajectories) from 10 to 100 causes *z~H~* dimensionality to scale up
> accordingly, while *z~L~* dimensionality remains stable. These results
> suggest an *emergent* separation of representational capacity between
> the modules that parallels their functional roles.
>
> To confirm that this hierarchical organization is an emergent property
> of training, and not an artifact of the network's architecture, we
> performed a control analysis using an identical but untrained network
> with random weights.
>
> We initialized an identical HRM architecture with random weights and,
> without any training, mea- sured the PR of its modules as the network
> processed the same task-specific inputs given to the trained model.
>
> The results, shown in Figure [8](#_bookmark11)(e,f), reveal a stark
> contrast: the high-level and low-level modules of the untrained
> network exhibit no hierarchical separation, with their PR values
> remaining low and nearly indistinguishable from each other. This
> control analysis validates that the dimensionality hierarchy is an
> *emergent property* that arises as the model learns to perform complex
> reasoning.
>
> The high-to-low PR ratio in HRM (*z~H~/z~L~ ≈* 2*.*98) closely matches
> that measured in the mouse cortex (*≈* 2*.*25). In contrast,
> conventional deep networks often exhibit *neural collapse*, where
> last-layer features converge to a low-dimensional subspace^80,81,82^.
> HRM therefore departs from the collapse pattern and instead fosters a
> high-dimensional representation in its higher module. This is
> significant because such representations are considered crucial for
> cognitive flexibility and are a hallmark of higher-order brain regions
> like the prefrontal cortex (PFC), which is central to complex
> reasoning.
>
> This structural parallel suggests the model has discovered a
> fundamental organizational principle. By learning to partition its
> representations into a high-capacity, high-dimensional subspace
> (*z~H~*)
>
> and a more specialized, low-dimensional one (*z~L~*), HRM autonomously
> discovers an organizational principle that is thought to be
> fundamental for achieving robust and flexible reasoning in biological
> systems. This provides a potential mechanistic explanation for the
> model's success on complex, long-horizon tasks that are intractable
> for models lacking such a differentiated internal structure. We
> emphasize, however, that this evidence is correlational. While a
> causal link could be tested via intervention (e.g., by constraining
> the H-module's dimensionality), such methods are difficult to
> interpret in deep learning due to potential confounding effects on the
> training process itself. Thus, the causal necessity of this emergent
> hierarchy remains an important question for future investigation.

# Related Work

> **Reasoning and algorithm learning** Given the central role of
> reasoning problems and their close relation to algorithms, researchers
> have long explored neural architectures that enable algorithm learning
> from training instances. This line of work includes Neural Turing
> Machines (NTM)^83^, the Differentiable Neural Computer (DNC)^84^, and
> Neural GPUs^85^--all of which construct iterative neural architectures
> that mimic computational hardware for algorithm execution, and are
> trained to learn algorithms from data. Another notable work in this
> area is Recurrent Relational Networks (RRN)^62^, which executes
> algorithms on graph representations through graph neural networks.
>
> Recent studies have integrated algorithm learning approaches with
> Transformer-based architec- tures. Universal Transformers extend the
> standard Transformer model by introducing a recurrent loop over the
> layers and implementing an adaptive halting mechanism. Geiping et al.
> ^86^ demonstrate that looped Transformers can generalize to a larger
> number of recurrent steps during inference than what they were trained
> on. Shen et al. ^16^ propose adding continuous recurrent reasoning
> tokens to the Transformer. Finally, TransNAR^8^ combine recurrent
> graph neural networks with language models.
>
> Building on the success of CoT-based reasoning, a line of work have
> introduced fine-tuning meth- ods that use reasoning paths from search
> algorithms (like A\*) as SFT targets^87,71,70^.
>
> We also mention adaptive halting mechanisms designed to allocate
> additional computational re- sources to more challenging problems.
> This includes the Adaptive Computation Time (ACT) for RNNs^88^ and
> follow-up research like PonderNet^89^, which aims to improve the
> stability of this allo- cation process.
>
> HRM further pushes the boundary of algorithm learning through a
> brain-inspired computational architecture that achieves exceptional
> data efficiency and model expressiveness, successfully dis- covering
> complex and diverse algorithms from just 1000 training examples.
>
> **Brain-inspired reasoning architectures** Developing a model with the
> reasoning power of the brain has long been a goal in brain-inspired
> computing. Spaun^90^ is one notable example, which uses spiking neural
> networks to create distinct modules corresponding to brain regions
> like the visual cortex and prefrontal cortex. This design enables an
> architecture to perform a range of cognitive tasks, from memory recall
> to simple reasoning puzzles. However, its reasoning relies on hand-
> designed algorithms, which may limit its ability to learn new tasks.
> Another significant model is the Tolman-Eichenbaum Machine (TEM)^91^,
> which is inspired by the hippocampal-entorhinal system's role in
> spatial and relational memory tasks. TEM proposes that medial
> entorhinal cells create a basis for structural knowledge, while
> hippocampal cells link this basis to sensory information. This
>
> allows TEM to generalize and explains the emergence of various cell
> types like grid, border, and place cells. Another approach involves
> neural sampling models^92^, which view the neural signaling process as
> inference over a distribution, functioning similarly to a Boltzmann
> machine. These models often require hand-made rules to be set up for
> solving a specific reasoning task. In essence, while prior models are
> restricted to simple reasoning problems, HRM is designed to solve
> complex tasks that are hard for even advanced LLMs, without
> pre-training or task-specific manual design.
>
> **Hierarchical memory** The hierarchical multi-timescale structure
> also plays an important role in how the brain processes memory. Models
> such as Hierarchical Sequential Models^93^ and Clockwork RNN^94^ use
> multiple recurrent modules that operate at varying time scales to more
> effectively cap- ture long-range dependencies within sequences,
> thereby mitigating the forgetting issue in RNNs.
>
> Similar mechanisms have also been adopted in linear attention methods
> for memorizing long con- texts (see the Discussions section). Since
> HRM focuses on reasoning, full attention is applied for simplicity.
> Incorporating hierarchical memory into HRM could be a promising future
> direction.

# Discussions

> **Turing-completeness of HRM** Like earlier neural reasoning
> algorithms including the Universal Transformer^95^, HRM is
> computationally universal when given sufficient memory and time con-
> straints. In other words, it falls into the category of models that
> can simulate any Turing machine, overcoming the computational
> limitations of standard Transformers discussed previously in the in-
> troduction. Given that earlier neural algorithm reasoners were trained
> as recurrent neural networks, they suffer from premature convergence
> and memory intensive BPTT. Therefore, in practice, their effective
> computational depth remains limited, though still deeper than that of
> a standard Trans- former. By resolving these two challenges and being
> equipped with adaptive computation, HRM could be trained on long
> reasoning processes, solve complex puzzles requiring intensive
> depth-first search and backtracking, and move closer to practical
> Turing-completeness.
>
> **Reinforcement learning with chain-of-thought** Beyond fine-tuning
> using human-annotated CoT, reinforcement learning (RL) represents
> another widely adopted training methodology. However, recent evidence
> suggests that RL primarily unlocks existing CoT-like capabilities
> rather than dis- covering fundamentally new reasoning
> mechanisms^96,97,98,99^. Additionally, CoT-training with RL is known
> for its instability and data inefficiency, often requiring extensive
> exploration and careful reward design. In contrast, HRM takes feedback
> from dense gradient-based supervision rather than relying on a sparse
> reward signal. Moreover, HRM operates naturally in a continuous space,
> which is biologically plausible and avoids allocating same
> computational resources to each token, even though tokens vary in
> their reasoning and planning complexity^16^.
>
> **Linear attention** Recurrence has been explored not only for its
> capability in universal computa- tion, but also as a means to replace
> the attention mechanism in Transformers, which suffers from quadratic
> time and memory complexity^100^. Recurrent alternatives offer a more
> efficient design by processing input tokens sequentially and
> predicting the next token at each time step, similar to early
> RNN-based language models.
>
> Some linear-attention variants, such as Log-linear Attention^101^,
> share an RNN-like state-update that can be interpreted as propagating
> multi-timescale summary statistics, thereby retaining long-range
> context without the quadratic memory growth of standard
> self-attention. However, substituting the attention mechanism alone
> does not change the fact that Transformers are still fixed-depth, and
>
> require CoT as a compensatory mechanism. Notably, linear attention can
> operate with a reduced key-value cache over extended contexts, making
> them more suitable for deployment on resource- constrained edge
> devices.

# Conclusion

> This work introduces the Hierarchical Reasoning Model, a
> brain-inspired architecture that lever- ages hierarchical structure
> and multi-timescale processing to achieve substantial computational
> depth without sacrificing training stability or efficiency. With only
> 27M parameters and train- ing on just 1000 examples, HRM effectively
> solves challenging reasoning problems such as ARC, Sudoku, and complex
> maze navigation--tasks that typically pose significant difficulties
> for contem- porary LLM and chain-of-thought models.
>
> Although the brain relies heavily on hierarchical structures to enable
> most cognitive processes, these concepts have largely remained
> confined to academic literature rather than being translated into
> practical applications. The prevailing AI approach continues to favor
> non-hierarchical models. Our results challenge this established
> paradigm and suggest that the Hierarchical Reasoning Model represents
> a viable alternative to the currently dominant chain-of-thought
> reasoning methods, ad- vancing toward a foundational framework capable
> of Turing-complete universal computation.
>
> **Acknowledgements** We thank Mingli Yuan, Ahmed Murtadha Hasan
> Mahyoub and Hengshuai Yao for their insightful discussions and
> valuable feedback throughout the course of this work.
