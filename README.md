# Paper
[IEEE Transactions on Sustainable Energy] Are Mechanisms Important for AI to Identify Oscillation Sources? A Case Study

## Abstract

Grid-connected wind turbine generators (WTGs) may induce sub-synchronous oscillations (SSOs) in a power system. Due to the difficulty to gain the detailed parameters of the WTGs in practice, data-driven AI method is considered to be a potential solution to identify the trouble-making WTGs (or SSO sources) in the power system. Unlike the SSO mechanism observed in traditional power systems, numerous studies and real-world SSO events have indicated that the SSO mechanisms in wind power grid-connected systems can be more complex and varied. However, most AI-based works ignore the fact that the SSO can arise from different mechanisms. Typically, AI models are trained and evaluated using data generated from a single type of SSO mechanism. However, in practice, AI models may need to identify the sources of SSOs caused by different or even unknown mechanisms. This raises an interesting question: Are mechanisms important for AI to identify oscillation sources? This paper preliminarily explores the answer to this question by study cases that consider two general SSO mechanisms: negative resistance and open-loop modal resonance. Further explainability analysis is carried out to investigate whether the SSO mechanisms affect the performance of the AI models. Results of study cases and explainability analysis provide researchers and engineers with deeper insights into the generalization ability of AI with respect to SSO mechanisms.

## Dataset Preparation
You can find our dataset for the Oscillation Source Localization here: [Google Drive](https://drive.google.com/drive/folders/1dGAl3Rb6wefdHkaHc-fYMajqBMuItwuE?usp=sharing).

## Result Verification
Main.ipynb

## Requirements

* Python 3.12
* PyTorch 2.7.0+cu128

## BibTeX Citation
If you find this paper and repository useful, please cite our paper 😊.
```bibtex
@article{xxx,
  title={xxx},
  author={xxx},
  journal={IEEE Transactions on Sustainable Energy},
  pages={xxx},
  year={xxx}
}
```
