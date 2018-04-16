Code to reproduce results in the paper "Defending against Adversarial Images using Basis Functions Transformations"

# Requirements

- Python 3.4 or higher
- TensorFlow 1.2.1
- Cleverhans 2.0
- sklearn
- matlab.engine

# Usage

We use Cleverhans to perform Fast Gradient Attack. 

## White-box attack

### Backward Pass Differentiable Approximation (BPDA)

```
python src/run_all_bpda.py
```

### Filtered Gradient Attack

```
python src/run_all_fga.py
```


