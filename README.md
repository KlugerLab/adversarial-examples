Code to reproduce results in the paper [Defending against Adversarial Images using Basis Functions Transformations](https://arxiv.org/pdf/1803.10840.pdf)

# Requirements

- Python 3.4 or higher
- TensorFlow 1.2.1
- Cleverhans 2.0
- sklearn
- matlab.engine

# Usage

We use Cleverhans to perform Fast Gradient Attack. 

## Gray-box attack 

Set self.setting = 'graybox' and run:

```
python run_all.py
```

You can modify which defense/attack methods to use by changing self.defense_list and self.attack_list in config.py.


## Black-box attack 

Set self.setting = 'blackbox' and run:

```
python run_all.py
```


## White-box attack

Set self.setting = 'whitebox'

### Backward Pass Differentiable Approximation (BPDA)

```
python src/run_all_bpda.py
```

### Filtered Gradient Attack

Set self.attack_list = ['FGA'] and run:

```
python src/run_all_fga.py
```


