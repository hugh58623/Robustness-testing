## Installation

#### 1. Using `pip` (recommended)

```bash
pip install OpenAttack
```

#### 2. Cloning this repo

```bash
git clone https://github.com/thunlp/OpenAttack.git
cd OpenAttack
python setup.py install
```

After installation, you can try running `demo.py` to check if OpenAttack works well:

```
python demo.py
```

## Create perturbed datasets [perturb] 
Go to /OpenAttack/examples/test.py, here is an example to use PWWS: "attacker = OpenAttack.attackers.PWWSAttacker()" . You can modify it as the attack models you want to use. You can check the attack models in Robustness-testing/OpenAttack/OpenAttack/attackers/. For example, if you want to use "viper", you can go to Robustness-testing/OpenAttack/OpenAttack/attackers/viper/__init__.py, and line 19 shows the "class VIPERAttacker(ClassificationAttacker):". You should modify as "attacker = OpenAttack.attackers.VIPERAttacker()" in /OpenAttack/examples/test.py.

Also you have to modify the  ori_file_path = 'path_to_dataset'.

Run test.py. You will get the perturbed dataset which end of "_adv.jsonl". For example, if your original file is "HumanEval.jsonl", the perturbed datsets will be "HumanEval_adv.jsonl"

## Attack Models

According to the level of perturbations imposed on original input, textual adversarial attack models can be categorized into sentence-level, word-level, character-level attack models. 

According to the accessibility to the victim model, textual adversarial attack models can be categorized into `gradient`-based, `score`-based, `decision`-based and `blind` attack models.

> [TAADPapers](https://github.com/thunlp/TAADpapers) is a paper list which summarizes almost all the papers concerning textual adversarial attack and defense. You can have a look at this list to find more attack models.

Currently OpenAttack includes 15 typical attack models against text classification models that cover **all** attack types. 

Here is the list of currently involved attack models.

- Sentence-level
  - (SEA) **Semantically Equivalent Adversarial Rules for Debugging NLP Models**. *Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin*. ACL 2018. `decision` [[pdf](https://aclweb.org/anthology/P18-1079)] [[code](https://github.com/marcotcr/sears)]
  - (SCPN) **Adversarial Example Generation with Syntactically Controlled Paraphrase Networks**. *Mohit Iyyer, John Wieting, Kevin Gimpel, Luke Zettlemoyer*. NAACL-HLT 2018. `blind` [[pdf](https://www.aclweb.org/anthology/N18-1170)] [[code&data](https://github.com/miyyer/scpn)]
  - (GAN) **Generating Natural Adversarial Examples**. *Zhengli Zhao, Dheeru Dua, Sameer Singh*. ICLR 2018. `decision` [[pdf](https://arxiv.org/pdf/1710.11342.pdf)] [[code](https://github.com/zhengliz/natural-adversary)]
- Word-level
  - (TextFooler) **Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment**. *Di Jin, Zhijing Jin, Joey Tianyi Zhou, Peter Szolovits*. AAAI-20. `score` [[pdf](https://arxiv.org/pdf/1907.11932v4)] [[code](https://github.com/wqj111186/TextFooler)]
  - (PWWS) **Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency**. *Shuhuai Ren, Yihe Deng, Kun He, Wanxiang Che*. ACL 2019. `score` [[pdf](https://www.aclweb.org/anthology/P19-1103.pdf)] [[code](https://github.com/JHL-HUST/PWWS/)]
  - (Genetic) **Generating Natural Language Adversarial Examples**. *Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, Kai-Wei Chang*. EMNLP 2018. `score` [[pdf](https://www.aclweb.org/anthology/D18-1316)] [[code](https://github.com/nesl/nlp_adversarial_examples)]
  - (SememePSO) **Word-level Textual Adversarial Attacking as Combinatorial Optimization**. *Yuan Zang, Fanchao Qi, Chenghao Yang, Zhiyuan Liu, Meng Zhang, Qun Liu and Maosong Sun*. ACL 2020. `score` [[pdf](https://www.aclweb.org/anthology/2020.acl-main.540.pdf)] [[code](https://github.com/thunlp/SememePSO-Attack)]
  - (BERT-ATTACK) **BERT-ATTACK: Adversarial Attack Against BERT Using BERT**. *Linyang Li, Ruotian Ma, Qipeng Guo, Xiangyang Xue, Xipeng Qiu*. EMNLP 2020. `score` [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.500.pdf)] [[code](https://github.com/LinyangLee/BERT-Attack)]
  - (BAE) **BAE: BERT-based Adversarial Examples for Text Classification**. *Siddhant Garg, Goutham Ramakrishnan. EMNLP 2020*. `score` [[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.498.pdf)] [[code](https://github.com/QData/TextAttack/blob/master/textattack/attack_recipes/bae_garg_2019.py)]
  - (FD) **Crafting Adversarial Input Sequences For Recurrent Neural Networks**. *Nicolas Papernot, Patrick McDaniel, Ananthram Swami, Richard Harang*. MILCOM 2016. `gradient` [[pdf](https://arxiv.org/pdf/1604.08275.pdf)]
- Word/Char-level
  - (TextBugger) **TEXTBUGGER: Generating Adversarial Text Against Real-world Applications**. *Jinfeng Li, Shouling Ji, Tianyu Du, Bo Li, Ting Wang*. NDSS 2019. `gradient` `score` [[pdf](https://arxiv.org/pdf/1812.05271.pdf)]
  - (UAT) **Universal Adversarial Triggers for Attacking and Analyzing NLP.** *Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, Sameer Singh*. EMNLP-IJCNLP 2019. `gradient` [[pdf](https://arxiv.org/pdf/1908.07125.pdf)] [[code](https://github.com/Eric-Wallace/universal-triggers)] [[website](http://www.ericswallace.com/triggers)]
  - (HotFlip) **HotFlip: White-Box Adversarial Examples for Text Classification**. *Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou*. ACL 2018. `gradient` [[pdf](https://www.aclweb.org/anthology/P18-2006)] [[code](https://github.com/AnyiRao/WordAdver)]
- Char-level
  - (VIPER) **Text Processing Like Humans Do: Visually Attacking and Shielding NLP Systems**. *Steffen Eger, Gözde Gül ¸Sahin, Andreas Rücklé, Ji-Ung Lee, Claudia Schulz, Mohsen Mesgar, Krishnkant Swarnkar, Edwin Simpson, Iryna Gurevych*. NAACL-HLT 2019. `score` [[pdf](https://www.aclweb.org/anthology/N19-1165)] [[code&data](https://github.com/UKPLab/naacl2019-like-humans-visual-attacks)]
  - (DeepWordBug) **Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers**. *Ji Gao, Jack Lanchantin, Mary Lou Soffa, Yanjun Qi*. IEEE SPW 2018. `score` [[pdf](https://ieeexplore.ieee.org/document/8424632)] [[code](https://github.com/QData/deepWordBug)]

The following table illustrates the comparison of the attack models.

|    Model    |  Accessibility  | Perturbation | Main Idea                                           |
| :---------: | :-------------: | :----------: | :-------------------------------------------------- |
|     SEA     |    Decision     |   Sentence   | Rule-based paraphrasing                             |
|    SCPN     |      Blind      |   Sentence   | Paraphrasing                                        |
|     GAN     |    Decision     |   Sentence   | Text generation by encoder-decoder                  |
| TextFooler  |      Score      |     Word     | Greedy word substitution                            |
|    PWWS     |      Score      |     Word     | Greedy word substitution                            |
|   Genetic   |      Score      |     Word     | Genetic algorithm-based word substitution           |
|  SememePSO  |      Score      |     Word     | Particle Swarm Optimization-based word substitution |
|  BERT-ATTACK  |      Score      |     Word     | Greedy contextualized word substitution |
|  BAE  |      Score      |     Word     | Greedy contextualized word substitution and insertion |
|     FD      |    Gradient     |     Word     | Gradient-based word substitution                    |
| TextBugger  | Gradient, Score |  Word+Char   | Greedy word substitution and character manipulation |
|     UAT     |    Gradient     |  Word, Char  | Gradient-based word or character manipulation       |
|   HotFlip   |    Gradient     |  Word, Char  | Gradient-based word or character substitution       |
|    VIPER    |      Blind      |     Char     | Visually similar character substitution             |
| DeepWordBug |      Score      |     Char     | Greedy character manipulation                       |

To know more information, please see the original project [OpenAttack](https://github.com/thunlp/OpenAttack/tree/master?tab=readme-ov-file).
