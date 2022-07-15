# DualAST-Pytorch
My pytorch implementation of the CVPR2021: DualAST. Code forked from pytorch implementation of AST(ECCV2018), which is also the foundation of DualAST.

### Intro
The biggest novelty of DualAST is **Dual**: two learning source of **Style**, collection of artwork from one artist (which control the overall style), and one specific artwork (which control the details of one painting, maybe some local pattern). SCB (style control block) is used for the specific artwork: pretrained vgg19 is used for extract style (for multi-layers).

#### train
`./model.py`
