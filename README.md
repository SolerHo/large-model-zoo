<h1 align="center"> 🏰 large model Zoo</h1>

- [Introduction](#introduction)
- [NLP models💬](#nlp-models)
- [CV models👀](#cv-models)
- [Multimodels](#multimodels)
- [TODO Lists 🚩](#todo-lists-)
- [Reference](#reference)


## Introduction
This project collects various of large-scale models as follows:
- NLP
- CV

Links to resource: Github, Paper, Hugging Face Etc.

All models are not sorted by any items, may be sorted by date or parameter size, etc.

## NLP models💬
|Model Name|Release Date|Developer/Institute|Size of Parameter|Github|Hugging Face|modelscope(魔搭)|Framework|Paper|Closed/Open source|
|--|--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Transformer|2017.06|Google|N / A|[[Link](https://github.com/huggingface/transformers)]|[[Link](https://huggingface.co/docs/transformers/index)]|[[Link](https://www.modelscope.cn/models/damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch/summary)]<br>(Alibaba DAMO)|----|[[Link](https://arxiv.org/pdf/1706.03762.pdf)]|Open|
|GPT 1.0|2018.06|OpenAI|117M|[[Link](https://github.com/openai/finetune-transformer-lm)]|[[Link](https://huggingface.co/openai-gpt)]|----|PyTorch|[[Link](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)]|Open|
|Bert|2018.10|Google|110M/340M|[[Link](https://github.com/google-research/bert)]|[[Link](https://huggingface.co/docs/transformers/model_doc/bert)]|[[Link](https://www.modelscope.cn/models/damo/nlp_bert_backbone_base_std/summary)]<br>(Alibaba DAMO)|TF|[[Link](https://aclanthology.org/N19-1423.pdf)]|Open||
|GPT-2|2019.02|OpenAI|124M/1158M|[[Link](https://github.com/openai/gpt-2)]|[[Link](https://huggingface.co/gpt2)]|----|PyTorch|[[Link](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)]|Open|
|XLNet|2019.06|CMU & Google|110M/240M|[[Link](https://github.com/zihangdai/xlnet)]|[[Link](https://huggingface.co/xlnet-base-cased)]|----|TF|[[Link](https://arxiv.org/pdf/1906.08237.pdf)]|Open|
|T5|2019.10|Google|60M/220M/770M|[[Link](https://github.com/google-research/text-to-text-transfer-transformer)]|[[Link](https://huggingface.co/docs/transformers/model_doc/t5)]|----|TF / JAX|[[Link](https://jmlr.org/papers/v21/20-074.html)]|Open|
|mT5|2020.10|Google|13B|[[Link](https://github.com/google-research/multilingual-t5/tree/master)]|[[Link](https://huggingface.co/docs/transformers/model_doc/mt5)]|----|TF|[[Link](https://arxiv.org/pdf/2010.11934.pdf)]|Open|
|GPT-3|2020.05|OpenAI|175B|[[Link](https://github.com/openai/gpt-3)]|----|----|PyTorch|[[Link](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)]|Closed|
|Pangu-Alpha|2020.07|Huawei & Peng Cheng Lab|2.6B|[[Link](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/PanGu-%CE%B1)]|[[Link](https://huggingface.co/imone/pangu_2_6B)]|[[Link](https://www.modelscope.cn/models/OpenICommunity/pangu_2_6B/summary)]|mindspore|[[Link](https://arxiv.org/pdf/2104.12369.pdf)]|Open|
|CPM-2|2021.06|Tsinghua University & BAAI(北京智源AI研究院)|11B/198B|[[Link](https://github.com/TsinghuaAI/CPM/tree/main)]|----|----|PyTorch|[[Link](https://arxiv.org/pdf/2106.10715.pdf)]|Open|
|T0|2021.03|Hugging Face|11B|[[Link](https://github.com/bigscience-workshop/t-zero)]|[[Link](https://huggingface.co/bigscience/T0)]|----|PyTorch|[[Link](https://arxiv.org/pdf/2110.08207.pdf)]|Open|
|PLUG|2021.04|Alibaba DAMO|27B|[[Link](https://github.com/alibaba/AliceMind/tree/main/PLUG)]|----|[[Link](https://www.modelscope.cn/models/damo/nlp_plug_text-generation_27B/summary)]|PyTorch|----|Open|
|Bloom|2021.08|Bloom|176B|----|[[Link](https://huggingface.co/bigscience/bloom)]|[[Link](https://modelscope.cn/models/langboat/bloom-2b5-zh/summary)]<br>(langboat Tech)|PyTorch|[[Link](https://arxiv.org/pdf/2211.05100.pdf)]|Closed|
|Codex (based on GPT3)|2021.07|OpenAI|----|----|----|----|----|[[Link](https://arxiv.org/pdf/2107.03374.pdf)]|Closed|
|LaMDA|2022.01|Google|2B|[[Link](https://github.com/conceptofmind/LaMDA-rlhf-pytorch)]|----|----|----|[[Link](https://arxiv.org/pdf/2201.08239v3.pdf)]|Open|
|OPT|2022.01|FaceBook(Meta)|125M ~ 175M|[[Link](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)]|[[Link](https://huggingface.co/docs/transformers/model_doc/opt)]|----|PyTorch|[[Link](https://arxiv.org/pdf/2205.01068.pdf)]|Closed|
|MT-NLG|2022.01|Microsoft|530B|----|----|----|PyTorch|[[Link](https://arxiv.org/abs/2201.11990)]|Closed|
|FLAN-T5v1.1|2021.09|Google|245B|[[Link](https://github.com/google-research/FLAN)]|[[Link](https://huggingface.co/docs/transformers/model_doc/flan-t5)]|----|TF|[[Link](https://arxiv.org/pdf/2109.01652v5.pdf)]|Open|
|LLaMA|2023.02|FaceBook(Meta)|7B ~ 65B|[[Link](https://github.com/facebookresearch/llama)]|[[Link](https://huggingface.co/docs/transformers/main/model_doc/llama)]|[[Link](https://modelscope.cn/models/Fengshenbang/Ziya-LLaMA-13B-v1.1/summary)<br>(Fengshenbang)]|PyTorch|[[Link](https://arxiv.org/pdf/2302.13971.pdf)]|Open|
|WebGPT|2021.12|OpenAI|175B|----|----|----|----|[[Link](https://arxiv.org/pdf/2112.09332.pdf)]|Closed|
|PaLM|2022.04|Google|540B|[[Link](https://github.com/lucidrains/PaLM-pytorch)]|----|----|PyTorch|[[Link](https://arxiv.org/pdf/2204.02311.pdf)]|Open|
|Gopher|2021.12|DeepMind|280B|----|----|----|----|[[Link](https://arxiv.org/pdf/2112.11446.pdf)]|Closed|
|PALM|2020.04|Alibaba DAMO|257M/483M|[[Link](https://github.com/alibaba/AliceMind/tree/main/PALM)]|----|[[Link](https://modelscope.cn/models/damo/nlp_palm2.0_text-generation_chinese-large/summary)]<br>(Alibaba DAMO)|PyTorch|[[Link](https://arxiv.org/pdf/2004.07159.pdf)]|Open|
|GPT-NeoX|2022.04|EleutherAI|20B|[[Link](https://github.com/EleutherAI/gpt-neox)]|[[Link](https://huggingface.co/docs/transformers/model_doc/gpt_neox)]|----|PyTorch|[[Link](https://arxiv.org/pdf/2204.06745.pdf)]|Open|
|AlphaCode|2021.01|DeepMind|----|----|----|----|----|[[Link](https://arxiv.org/abs/2203.07814)]|Closed|
|InstructGPT|2022.01|OpenAI|1.3B|----|----|----|----|[[Link](https://arxiv.org/pdf/2203.02155.pdf)]|Closed|
|CodeGen|2022.01|SaleForce Research|350M/1B/3B/7B/16B|[[Link](https://github.com/salesforce/CodeGen)]|[[Link](https://huggingface.co/docs/transformers/model_doc/codegen)]|----|PyTorch|[[Link](https://arxiv.org/pdf/2203.13474.pdf)]|Open|
||||||||||



## CV models👀
|Model Name|Release Date|Developer/Firms|Size of Parameter|Domain|Github|Hugging Face|Supported Framework|Paper|Closed / Open source|FLOPS|Top-1 Error|Top-5 Error|
|--|--|--|--|--|--|--|--|--|--|--|--|--|
|ResNet|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

## Multimodels

## TODO Lists 🚩
- [ ] NLP models
- [ ] CV models
- [ ] Hybrid models
- [ ] Other

## Reference
- LLM：https://arxiv.org/pdf/2303.18223.pdf
- https://www.datalearner.com/ai-models/pretrained-models?&aiArea=2&openSource=-1&publisher=-1
