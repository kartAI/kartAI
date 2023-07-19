# Guide for setting up the kartAI code

## Clone/download the code

You can choose to either:

- Clone the repo using git
  
  ```git clone https://github.com/kartAI/kartAI.git```

- Or download a zipped folder with the code:
  - click on the green "code" button, and choose "Download zip"

## Installing dependecies

Go to installation guide for your OS:

1. [Mac](#mac)
2. [Windows](#windows-with-wsl2)
3. [Linux](#linux)

### Mac

1. **System requirements**

    You need macOS 10.12.6 (Sierra) or higher (64-bit).

2. **Check Python version**

    Make sure you have Python 3.8–3.11, and pip >= 20.3.

    ```bash
    python3 --version
    python3 -m pip --version
    ```

3. **Install Miniconda**

    ```bash
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
    bash Miniconda3-latest-MacOSX-x86_64.sh
    ```

    You may need to restart your terminal or source ~/.bashrc to enable the conda command. Use conda -V to test if it is installed successfully

4. **Create environment with the needed dependencies**

    Make sure to move to the root folder of the kartAI project before running this code.

    ```bash
    conda create -f environments/mac-cpu-environment.yml
    ```

5. **Activate environment**

    In order to use the newly created environment you need to activate it

    ```bash
    conda activate kai-mac
    ```

### Windows with WSL2

1. **System requirements**

    Windows 10 19044 or higher (64-bit). This corresponds to Windows 10 version 21H2, the November 2021 update.

    See the following documents to:

    - [Download the latest Windows 10 update.](https://www.microsoft.com/software-download/windows10)
    - [Install WSL2](https://docs.microsoft.com/windows/wsl/install)

2. **Check Python version**

    Make sure you have Python 3.8–3.11, and pip >= 20.3.

    ```bash
    python3 --version
    python3 -m pip --version
    ```

3. **Install Miniconda**

    ```bash
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    ```

    You may need to restart your terminal or source ~/.bashrc to enable the conda command. Use conda -V to test if it is installed successfully

4. **Create environment with the needed dependencies**

    Make sure to move to the root folder of the kartAI project before running this code.

    ```bash
    conda create -f environments/wsl2-cpu-environment.yml
    ```

5. **Activate environment**

    In order to use the newly created environment you need to activate it

    ```bash
    conda activate kai-wsl2
    ```

### Linux

1. **System requirements**

    Ubuntu 16.04 or higher (64-bit)

2. **Check Python version**

    Make sure you have Python 3.8–3.11, and pip >= 20.3.

    ```bash
    python3 --version
    python3 -m pip --version
    ```

3. **Install Miniconda**

    ```bash
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    ```

    You may need to restart your terminal or source ~/.bashrc to enable the conda command. Use conda -V to test if it is installed successfully

4. **Create environment with the needed dependencies**

    Make sure to move to the root folder of the kartAI project before running this code. 

    ```bash
    conda create -f environments/linux-cpu-environment.yml
    ```

5. **Activate environment**

    In order to use the newly created environment you need to activate it

    ```bash
    conda activate kai-linux
    ```
