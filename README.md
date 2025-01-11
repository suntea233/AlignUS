# AlignUS: Utterance Alignment of Language Models for Effective User Simulation in Task-Oriented Dialogues

The official Pytorch implementation of AlignUS (TASLP 2025)

![image](https://github.com/suntea233/AlignUS/blob/main/architecture.png)
<p align="center"><em> Overall architecture </em>

## 🔥 Run our Code

Create a new environment with python==3.9
```shell
conda create -n alignus python==3.9
```

Train the BART model
```shell
python train_model.py
```

After training the BART model, using the [ConvLab-3](https://github.com/ConvLab/ConvLab-3) framework to generate post-training data with LLMs, the LLMs are constructed in **AlignUS.py** and **base_model.py**.
Then change the training data and checkpoint for the BART model.
Start to Post-train the BART model
```shell
python train_model.py
```

Test on the platform provided by ConvLab-3.
```shell
python script.py
```

### 👏 Credits
*A shout-out to the authors of [ConvLab-3](https://github.com/ConvLab/ConvLab-3) and [GenTUS](https://aclanthology.org/2022.sigdial-1.28/) for building the framework from which we constructed the foundations of this work.*
