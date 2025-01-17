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

> Install from requirements file in src/environment
```
pip install -r ./environment/requirements.txt
```

(This will install necessary packages including dash)

## Initialize GEE 
> Install gcloud
>> Download and Install gcloud from `https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe` or alternatively run the following command in PowerShell terminal;

`(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")`

`& $env:Temp\GoogleCloudSDKInstaller.exe`

>> After installation is complete, the installer gives you the option to create Start Menu and Desktop shortcuts, start the Google Cloud CLI shell, and configure the gcloud CLI. Make sure that you leave the options to start the shell and configure your installation selected. 

>> The installer will start a terminal window and runs the gcloud init command where you will be prompted to select your default project.
    
> From the terminal run the following command;

`earthengine authenticate --auth_mode=gcloud --force`

>> This command will open a new browser window or tab and prompt you to log in with your Google account. Follow the instructions to authenticate your account. After successful authentication, you'll receive a token to paste back into your terminal, which links your GEE account with your development environment.

>> For additional guidelines, refer to `https://developers.google.com/earth-engine/guides/auth`

## Test GEE correct functionality
> A test file `test_gee.py` has been included in the src/utils directory, run the following command to test that the installations and initialization of GEE is correctly done. You should get a response similar to this;

```
{'type': 'Image', 'bands': [{'id': 'elevation', 'data_type': {'type': 'PixelType', 'precision': 'int', ... }}]}
```

> ***Note:*** You may be redirected to your GCP to enable the GEE API if not yet active.