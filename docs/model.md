# The model

This page describes what is simulated, with the equations and default
parameters. Everything lives in `wetware.network`, `wetware.neurons`,
`wetware.recorder` and `wetware.culture`, and every parameter mentioned here is
a field of `NetworkParams`, `RecorderParams`, `StimulationParams` or
`AgingParams`.

It is a phenomenological model. It is tuned to look like a dissociated
cortical culture on an MEA at the level of spike rasters, not fitted to any
dataset.

## Geometry

`n_neurons` cells are placed uniformly at random on a rectangle that extends
`margin_um` (150 µm) past the outer electrodes. 80 % are excitatory
(`exc_fraction`). With the default 8x8 array at 200 µm pitch and 1000 neurons
the density is about 350 cells/mm², which is sparse compared to a real
culture; raise `n_neurons` if you need more units per electrode.

## Neurons

The default is the Izhikevich (2003) simple model:

$$
\dot v = 0.04v^2 + 5v + 140 - u + I, \qquad \dot u = a(bv - u)
$$

with reset $v \leftarrow c,\; u \leftarrow u + d$ when $v \ge 30$ mV.
Excitatory cells are regular spiking with a random shift towards chattering
($a = 0.02$, $b = 0.2$, $c = -65 + 15r^2$, $d = 8 - 6r^2$, $r \sim U(0,1)$);
inhibitory cells are fast spiking ($a = 0.02 + 0.08r$, $b = 0.25 - 0.05r$,
$c = -65$, $d = 2$). Integration is forward Euler at `dt_ms` (0.5 ms).

A leaky integrate-and-fire model with spike-frequency adaptation is also
built in (`neuron_model="lif"`):

$$
\tau_m \dot v = -(v - v_\text{rest}) + R\,(I - w), \qquad \tau_w \dot w = -w
$$

with $w \leftarrow w + b$ at each spike and a 2 ms refractory period. Network
defaults are tuned for Izhikevich; the LIF culture is quieter and bursts less.
Custom models implement `wetware.neurons.NeuronModel` (see the API page).

## Connectivity

Neuron $i$ connects to neuron $j$ with probability

$$
p_{ij} = p_\text{max}\, e^{-d_{ij}/\lambda}
$$

(`p_connect_max` = 0.35, `connect_lambda_um` = 250 µm), without autapses.
Weights are log-normal around `w_exc` = 2.5 and `w_inh` = -4 ($\sigma$ = 0.5),
then scaled by $60 / \langle k_\text{in} \rangle$ so that the total input stays
comparable when you change the network size.

## Synapses and short-term plasticity

A spike adds to an exponentially decaying current on each target
($\tau_E$ = 5 ms, $\tau_I$ = 10 ms). The amount is the weight times a
Tsodyks-Markram release factor kept per presynaptic cell:

$$
u \leftarrow u + U(1-u), \qquad \text{release} = u\,x, \qquad x \leftarrow x - u\,x
$$

$$
\dot x = \frac{1 - x}{\tau_\text{rec}}, \qquad \dot u = \frac{U - u}{\tau_\text{facil}}
$$

Excitatory synapses are strongly depressing ($U$ = 0.5, $\tau_\text{rec}$ = 4 s,
$\tau_\text{facil}$ = 50 ms); inhibitory ones mildly facilitating ($U$ = 0.2,
$\tau_\text{rec}$ = 200 ms, $\tau_\text{facil}$ = 400 ms). Depression of
recurrent excitation is what terminates a network burst and sets the interval
to the next one.

## STDP

Excitatory synapses follow pair-based STDP with exponential traces
($\tau_\pm$ = 20 ms):

$$
\Delta w_{ij} = +A_+\, w_\text{max}\, \bar x_i \;\; \text{(post spike)}, \qquad
\Delta w_{ij} = -A_-\, w_\text{max}\, \bar y_j \;\; \text{(pre spike)}
$$

with $A_+$ = 0.010, $A_-$ = 0.0105 and hard bounds $[0, w_\text{max}]$.
Traces below $10^{-3}$ are treated as zero to keep the update sparse. Turn it
off with `Culture(plasticity=False)` or `culture.plasticity = False`.

## Background drive and bursts

Each neuron gets a constant bias $\sim \mathcal N(0, 1)$ plus white noise
(`noise_sd` = 2, scaled by $1/\sqrt{dt}$). 3 % of excitatory cells are
pacemakers with a bias of about 4.5, enough to fire tonically. Their activity
occasionally recruits the recurrent network into a network burst; depression
then silences it. With defaults a 1000-neuron culture produces roughly one
network burst per second on top of ~1 Hz background firing per neuron.

## Stimulation

Stimulation is current-controlled. A pulse with effective (cathodic) charge
$Q$ on electrode $e$ depolarizes neuron $i$ by

$$
\Delta v_i = g\, Q\, \frac{\epsilon_i\, \xi}{1 + (d_{ei}/r_0)^2}
$$

with $g$ = 30 mV/nC, $r_0$ = 90 µm, a fixed per-neuron excitability
$\epsilon_i$ (log-normal, $\sigma$ = 0.3), per-trial jitter $\xi$
($\sigma$ = 0.2), and no effect beyond 400 µm. Pulse shape beyond its charge
is ignored.

Safety limits (`SafetyLimits`) are checked before anything is scheduled and
raise `StimulationSafetyError`:

| Limit | Default |
|---|---|
| phase amplitude | 50 µA |
| phase duration | 1000 µs |
| charge per phase | 10 nC |
| charge density per phase | 2000 µC/cm² |
| charge balance | net charge within 1 % of the phase charge |
| pulse rate per electrode | 200 Hz |
| electrodes per pulse | 16 |

The defaults are placeholders, not material limits. Set them to match your
real hardware.

## Recording chain

1. **Spike waveforms.** A spike of neuron $i$ adds a ~2 ms biphasic template
   to electrode $e$ with amplitude $A_i / (1 + (d_{ei}/30\,\mu m)^2)$,
   $A_i \approx 150$ µV (log-normal), and nothing beyond 150 µm.
2. **Noise.** White Gaussian noise per electrode, 5 µV RMS on average with
   20 % spread between electrodes.
3. **Stimulation artifact.** The amplifier saturates for the length of the
   pulse, then the artifact decays with $\tau$ = 0.4 ms. It reaches other
   electrodes with a Lorentzian falloff (300 µm).
4. **Blanking.** Samples within `blank_ms` (3 ms) after each pulse are zeroed
   on every electrode, so direct responses in that window are lost.
5. **Detection.** Negative threshold crossings at $-5\sigma$, where
   $\sigma = \text{median}(|x|)/0.6745$ (Quiroga et al., 2004), with a 1 ms
   dead time. The threshold is re-estimated on every chunk of 100 ms or more.

Traces are sampled at 10 kHz by default.

## Aging

`culture.age(days)` moves the culture forward in days-in-vitro (DIV) without
simulating the activity in between:

- **Maturation.** Synaptic strength follows a logistic curve centred on DIV 18
  (width 3 days), so a DIV 7 culture is almost unconnected.
- **Decline.** After DIV 35 the background drive decays with a 45-day time
  constant and weights with a 120-day time constant: firing rates fall and
  bursts become rarer and smaller.
- **Cell death.** Each neuron dies with probability 0.2 %/day, rising by
  0.04 %/day per day past DIV 35.
- **Electrode dropout.** Each electrode fails with probability 0.2 %/day,
  rising by 0.02 %/day per day past DIV 35. Dead electrodes record a flat
  trace and reject stimulation.

## What is not modelled

Axonal conduction delays, dendrites, NMDA/GABA-B kinetics, homeostatic
plasticity, structural plasticity, glia, the medium, temperature, 1/f and
line noise, electrode impedance, and spike sorting. Detected spikes are
multi-unit activity per electrode.

## References

- Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE
  Transactions on Neural Networks*, 14(6), 1569-1572.
- Tsodyks, M., Pawelzik, K., & Markram, H. (1998). Neural networks with
  dynamic synapses. *Neural Computation*, 10(4), 821-835.
- Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian
  learning through spike-timing-dependent synaptic plasticity. *Nature
  Neuroscience*, 3(9), 919-926.
- Quiroga, R. Q., Nadasdy, Z., & Ben-Shaul, Y. (2004). Unsupervised spike
  detection and sorting with wavelets and superparamagnetic clustering.
  *Neural Computation*, 16(8), 1661-1687.
- Wagenaar, D. A., Pine, J., & Potter, S. M. (2006). An extremely rich
  repertoire of bursting patterns during the development of cortical
  cultures. *BMC Neuroscience*, 7, 11.
- Kagan, B. J., et al. (2022). In vitro neurons learn and exhibit sentience
  when embodied in a simulated game-world. *Neuron*, 110(23), 3952-3969.
- Jordan, F. D., et al. (2024). Open and remotely accessible Neuroplatform
  for research in wetware computing. *Frontiers in Artificial Intelligence*,
  7, 1376042.
