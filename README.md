# Setup
## 1. Change some strings

1. Anaconda folder

```json
// ./vscode/settings.json
{
    "the_anaconda_folder": "/home/sunao/anaconda3"
}
```

2. Hugging face token

```Makefile
// makefile
<YOUR_HUGGINGFACE_USER>
<YOUR_HUGGINGFACE_TOKEN>
```

## 2. Run makefile

to init conda env / deps / dataset / models

```
make init-git 
make init-conda-eval
make download_datasets
```


## 3. Open vscode and debug!

