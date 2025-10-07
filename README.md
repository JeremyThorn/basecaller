# Learning Basecaller Project

The purpose of this project is to allow me investigate the techniques used for calling nucleobases from current traces output by nanopore-type devices. I have no access to GPUs, and so the project runs on the CPU. The current models exposed are a simple 1d-conv only classifier, and a larger encoder-BiGRU classifier. Both models work in an offline fashion. It would be interesting to expand the scope to include online classifiers and transformer models.

## Install Dependencies

Use

```bash
python3 -m venv venv
source venv/bin/activate
```

to create and activate an environment. Now, run

```bash
python -m pip install -U pip
python -m pip install -r requirements.in
```

to install the project dependencies.

## Generate data and run training

We must first generate a synthetic dataset on which to learn. The model used generates current distributions for each k-mer as follows.

Let $k$ be odd with center index $c=\lfloor k/2 \rfloor$.
For a k-mer $\mathbf{b}=(b_0,\dots,b_{k-1})$, the **mean** current is

$$
\mu(\mathbf b) = I_0 + L_c\big(b_c\big) + \sum_{p \ne c} F_p\big(b_p\big) + \sum_{p=0}^{k-2} P_p\big(b_p, b_{p+1}\big) + \big(u_{sd}(\mathbf b) - 1/2\big)\delta,
$$

and the per-k-mer **standard deviation** is

$$
\sigma(\mathbf b)
= sd_{\mathrm{lo}} + \big(sd_{\mathrm{hi}} - sd_{\mathrm{lo}}\big) u_{sd}(\mathbf b).
$$

Where:

- $I_0$ is the **current center** (``current_center``).
- $L_c(\cdot)$ are **center levels**: equally spaced, zero-mean levels deterministically assigned to bases and scaled by ``pos_scale_center``.
- $F_p(\cdot)$ are **flank position contributions** for $p \ne c$: deterministic $(u-0.5)\cdot$ ``pos_scale_flank``.
- $P_p(\cdot,\cdot)$ are **adjacent pair contributions**: deterministic $(u-0.5)\cdot$ ``pair_scale``.
- $u_{sd}(\mathbf b)\in[0,1)$ is a deterministic hash-based pseudo-uniform for the k-mer.
- Optional $\delta$, controlled by ``global_jitter`` adds a small zero-mean perturbation to the mean.

To generate the synthetic reads, run

```bash
python scripts/generate_reads.py
```

The raw data must now be packaged into learnable windows that the model will operate on. To do this, run

```bash
python scripts/prepare_dataset.py
```

Finally, we are ready to train a model. To do this, run

```bash
python scripts/run_train.py
```

The model will begin training. This takes a few hours on the CPU, so a pre-trained model is provided for evaluation. This model obtains a character error rate of around 9\%. To evaluate this model, run

```bash
python scripts/evaluate.py
```
