# Setup Guide for GCP VM

1. Create a new project on [GCP Cloud Platform](https://console.cloud.google.com)
2. Add a VM with the desired configuration (request a quota to be able to GPU support). Use the special disk image `[c0-deeplearning-common-gpu-v20231105-debian-11-py310` to get 50GB of disk space and `conda` pre-installed with the correct Python version.
3. Use the VM by using the SSH-in-browser feature by GCP

**Local SSH**
To access the VM from the local terminal, follow these steps

Generate a new key pair with the following command and copy the public key into the clipboard

```bash
ssh-keygen -t rsa -f ~/.ssh/<keyname>
cat ~/.ssh/<keyname>.pub | pbcopy
```

Log into the VM via ssh-in-browser and append the public key to the file `~/.ssh/authorized_keys`

```bash
echo "<public-key>" >> ~/.ssh/authorized_keys
```

Find the *username* and *external IP* address for the VM from the Google Cloud Platform.

```bash
ssh -i ~/.ssh/<private-key> <user>@<external_ip>
```

*Note: Commands that are to be run be run by sudo.*

**Git**
Install git via `apt-get`

```bash
sudo apt-get update
sudo apt-get install git-all
```

*Check that installation was successful via version: `git --version`*

Set global git configuration using

```bash
git config --global user.name "mikasenghaas"
git config --global user.email "mail@mikasenghaas.de"
```

Clone the repository using

```bash
git clone https://github.com/...
```

*This will prompt for user authentication. Provide a authentication token.*

**Conda**
*NB: If the VM instance was created from a pre-set image specifically for deep learning, it comes pre-installed with conda. In that case, it is easier to use create a conda environment.*

Navigate to the cloned directory and create a virtual environment

```bash
cd /path/to/project
conda env create -f environment.yml
```

This will create a virtual environment as specified in the config file and resolve all dependencies. To activate the environment

```bash
conda activate <name>
```

To deactivate the environment simply type `deactivate`.

**Python**
*NB: Follow this if `conda` is not pre-installed.*

Install a supported version of Python

```bash
sudo apt install python3 python3-dev python3-venv
```

Install `pip` to install packages inside the virtualenv

```bash
sudo apt-get install wget
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
```

*Verify that `pip` was installed by running `pip3 --version*

Navigate to the cloned directory and create a virtual environment

```bash
cd /path/to/project
python3 -m venv env
```

Activate the virtual environment

```bash
source env/bin/activate
```

Install all dependencies

```bash
pip install -r requirements.txt
```

**Transfer data**
Due to memory constraints on the VM, we preprocess the data on the local machines and then copy the data to the VM.
As per this [guide](https://cloud.google.com/compute/docs/instances/transfer-files#scp), there are multiple ways to transfer files to the VM. We use `scp` to transfer the data.

```bash
scp -i ~/.ssh/<private-key> -r /local/path <user>@<external_ip>:~/remote/path
```

We do this both for the data in `data/swissprot/processed` and `data/tabula_muris/processed`.
