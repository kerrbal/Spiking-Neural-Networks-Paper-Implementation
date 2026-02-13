# Surrogate Gradient Learning in Spiking Neural Networks

I implemented Surrogate Gradient Learning in Spiking Neural Networks paper in this project.
SNNs are a wide topic, so I implemented two approaches. In the first one, I implemented LIF cells, and I used random weight matrix to project
delta values from output layer into the hidden layer. It ensures the neurons are isolated in the model as in our brain. I used SuperSpike as surrogate gradients to smooth the voltage gradient whose formula is like -> ∂u∂s​≈(1+β∣u−θ∣)^2​. I trained MNIST dataset in the SNN_models.ipynb code file. Mnist dataset is a static dataset, so I converted it using Poisson Encoding, and Rate Encoding (approximate performances for each encoding style). Also for my first model, you must set last_learning_window_size variable. It is used for taking the loss values and gradients for just the last time steps. It is useful in event-based datasets because in the first time-steps may lack information, so penalizing weights is not sensible.

In my second model, I implemented trace. Biologically our brains leave out traces from previous time steps, which is used for changing the connection strengh (training). It holds the temporal context of the dataset, however there is a exploding/vanishing gradient problem as well which needs to be taken care of.

You can see the trace implementation on the Trace.py file.

You can run DVS_Gestures.ipynb file for seeing the performance of the model in event-based datasets. Note: the dataset is around 5GB.  

## LIF Neuron Model

The LIF neuron dynamics are defined as:

$$
I(t+1) = B(t) + S(t)W^{\top}
$$

$$
U(t+1) = \alpha U(t) + I(t) - S(t)\theta
$$

## Eligibility Trace Model

The synaptic eligibility trace is defined as:

$$
e_{ij}(t) = \lambda \, e_{ij}(t-1) + s_i^{\text{pre}}(t) \, \frac{\partial s_j^{\text{post}}(t)}{\partial u_j(t)}
$$

The weight update rule follows a three-factor learning rule:

$$
\Delta w_{ij} = -\eta \sum_t \delta_j(t) \, e_{ij}(t)
$$
