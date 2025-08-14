#import "@preview/ilm:1.4.1": *
#import "@preview/diverential:0.2.0": *

#set text(lang: "en")

#show: ilm.with(
  title: [Neural Network Model for Alzheimer's Disease],
  author: "Yilin Li",
  date: datetime(year: 2025, month: 08, day: 14),
  abstract: [
    In this note, we would illustrate the progression on ADNN populational model. More specifically, focus on the model structure and their behaviors.
  ],

  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
)


The main focus is to solve the problem that the trajectory function diverges at both ends and does not exhibit an S-shaped curve.

This attempt explored four architectures:

= PINN + FNN

The neural network is an ordinary FNN with two hidden layers, using Tanh or ReLU as activation at first, then Sigmoid.

Specifically, the loss applies the idea of PINN: encourage the ends of the trajectory to be smooth ($f$ approaches 0 when $||input|| to y_5, y_95$) and the gradient in the middle to be as large as possible (f larger when $||input|| to 0$, resulting in smaller loss).

More specifically, the loss function is defined as:
$ L = (L_{global} + Residual) * exp(||f(y_5, s_default)|| + ||f(y_95, s_default)||) / (||f(y_0, s_default)|| + 1e-5) $

$L_{global}$ was introduced in the previous document — it is the loss of the entire trajectory for a specific patient.  
$Residual$ is the residual loss.  

In addition, a *product-form* PINN was attempted:  
dy/ds = y(1 - y) f(y, s)  
(I did not directly replace the left polynomial with the actual polynomial model because it would be meaningless.)

Directly using dy/ds = y(1 - y) f(y, s) produced highly unstable results, prone to explosion.

To address this, a sigmoid constraint was used:  
$ d y/d s = sigma(y) (A - sigma(y)) f(y, s) $,  
where $ A in R^4 $ represents the upper limits of each biomarker in logistic regression.

The results from the above approaches were consistent: the network learned a straight or nearly straight line.

= GRU-RNN

A gated neural network, with the structure as follows:  
(figure placeholder)

This structure can learn a small amount of curvature but not an S-shaped curve.

= LSTM

An evolved version of GRU-RNN, with similar results.

The specific network structure was not saved to GIT.

= Residual Neural Network + Fine Tuning

Results were similar; structure was not saved.

= Ideas for Further Improvement

Mainly based on other related work, especially discussions on activation functions.

== 1. Multi-head + Classification Model

Jung et al., 2021, MedIA — *Deep Recurrent Model for Individualized Trajectories*

Goal: Jointly model longitudinal continuous measures and discrete diagnosis for individuals, capable of handling irregular sampling/missing data.

Core algorithm: embed disease progression into latent state $ z_t $ and update it with RNN:

$ z_t = f_theta(z_{t-1}, x_t, Delta t_t) $

Multi-head decoding predicts continuous variables (Gaussian likelihood / MSE) and classification (cross-entropy); missingness is explicitly modeled via masks/time intervals into $ f_theta $.  
The paper provides a complete objective and optimization from Eq. (2) to Eq. (12).  
Implemented as RNN/GRU/LSTM backbone + multi-task heads; activations tanh/sigmoid/softmax; authors provide GitHub (model.py etc.).

== 2. BINN

dx/dt = f_known(x) + g_NN(x)  
Here, f_known(x) comes from a classical ODE system in the literature.

Need to read the paper to see what results generally make sense.


