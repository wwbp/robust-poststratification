# Robust Post-Statification

## Setup

Install following dependencies
* python3
* pip

(Optional) Create and activate a virtual environment

```sh
python3 -m venv venv
source venv/bin/activate
```

Install dependencies

```sh
pip install -r requirements.txt
```

User table, population table, and output must be specified in the arguments. The demographics argument must be valid columns in the user table.

Script can be run using any combination of the following
* multiple demographics (raking or naive post-stratification)
* redistribution
* smooth before binning
* uninformed smoothing (ignores smoothing_k)

## Examples

Single correction factor (income)

```sh
python3 robust_poststrat.py --demographics income --smoothing_k 10 --mininum_bin_threshold 50 --user_table /path/to/user_table.csv --population_table /path/to/population_table.csv --output /path/to/output.csv
```

Single correction factor (income) with redistribution

```sh
python3 robust_poststrat.py --demographics income --smoothing_k 10 --mininum_bin_threshold 50 --redistribution --user_table /path/to/user_table.csv --population_table /path/to/population_table.csv --output /path/to/output.csv
```

Single correction factor (income) with smooth before binning

```sh
python3 robust_poststrat.py --demographics income --smoothing_k 10 --mininum_bin_threshold 50 --smooth_before_binning --user_table /path/to/user_table.csv --population_table /path/to/population_table.csv --output /path/to/output.csv
```

Single correction factor (income) with uninformed smoothing (smoothing_k is ignored)

```sh
python3 robust_poststrat.py --demographics income --smoothing_k 10 --mininum_bin_threshold 50 --uninformed_smoothing --user_table /path/to/user_table.csv --population_table /path/to/population_table.csv --output /path/to/output.csv
```

Multiple correction factors (income + education) using raking

```sh
python3 robust_poststrat.py --demographics income education --smoothing_k 10 --mininum_bin_threshold 50 --user_table /path/to/user_table.csv --population_table /path/to/population_table.csv --output /path/to/output.csv
```

Multiple correction factors (income + education) using naive post-stratification

```sh
python3 robust_poststrat.py --demographics income education --smoothing_k 10 --mininum_bin_threshold 50 --naive_poststrat --user_table /path/to/user_table.csv --population_table /path/to/population_table.csv --output /path/to/output.csv
```

Multiple correction factors (age + gender + income + education) using raking with redistribution

```sh
python3 robust_poststrat.py --demographics age gender income education --smoothing_k 10 --mininum_bin_threshold 50 --redistribution --user_table /path/to/user_table.csv --population_table /path/to/population_table.csv --output /path/to/output.csv
```

## Input and Output Format

## Dependencies

* pandas
* numpy
* [quantipy3](https://github.com/Quantipy/quantipy3)

## Citation

If you use this code in your work please cite the following paper:

```
@article{giorgi2022correcting,
      title={Correcting Sociodemographic Selection Biases for Population Prediction from Social Media}, 
      author={Salvatore Giorgi and Veronica Lynn and Keshav Gupta and Farhan Ahmed and Sandra Matz and Lyle Ungar and H. Andrew Schwartz},
      year={2022},
      journal={Proceedings of the International AAAI Conference on Web and Social Media}, 
}
```