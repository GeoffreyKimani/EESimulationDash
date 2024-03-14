# Simple dash app to demo Yiled Prediction
> The dash app is created from the initial [notebook](https://colab.research.google.com/drive/1MMTQIwuXv9f2q_W257L6LrLpIll0d34B?usp=sharing) based on dataset from GiZ competition and funding of one Project headed by Ines and Dan

## Installation 
> Create virtual env

```
python -m venv venv
```

> Activate the env

```
. venv/bin/activate (Linux/MacOs)
. .\venv\Scripts\activate (Windows)
```

> Install from requirements file
```
pip install -r requirements.txt
```

(This will install necessary packages including dash)

## Initialize GEE 
> From the terminal run the following command;

`earthengine authenticate`

>> This command will open a new browser window or tab and prompt you to log in with your Google account. Follow the instructions to authenticate your account. After successful authentication, you'll receive a token to paste back into your terminal, which links your GEE account with your development environment.

## Test GEE correct functionality
> A test file `test_gee.py` has been included in the root directory, run the following command to test that the installations and initialization of GEE is correctly done. You should get a response similar to this;

```
{'type': 'Image', 'bands': [{'id': 'elevation', 'data_type': {'type': 'PixelType', 'precision': 'int', ... }}]}
```

> ***Note:*** You may be redirected to your GCP to enable the GEE API if not yet active.