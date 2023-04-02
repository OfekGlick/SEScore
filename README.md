
# SEScore

## Description
In this repo we explore different methods to improve the already great SEScore evaluation metric.<br>

### Background
SEScore is a reference-based text-generation evaluation metric that requires no pre-human-annotated error data, described in the paper [Not All Errors are Equal: Learning Text Generation Metrics using Stratified Error Synthesis.](https://arxiv.org/abs/2210.05035)<br>
Generally speaking, the paper describes a stratified dataset synthesis pipeline where sentences get corrupted via pre-defined methods, then the newly corrupted sentences receive a score that represends how "severe" the corruption was via bi-directional entailment and finally we train a NN to learn the scores accumulated by the bi-directional entailment model. <br>
While this method performs very well and has improved upon the SOTA, we believe there is room for improvement.

### Suggested improvements
1. The paper describes a stratified way to accumulated errors via corruption of the sentences. The corruption of the sentences occurs by Adding/Replacing/Deleting/Swapping tokens in the original sentence. While effective, recent papers showed more effective masking techniques which could help create more meaningfull corruption. Our first proposal would be to use [PMI masking](https://arxiv.org/abs/2010.01825) instead of token masking in the corruption of the sententces.
2. The severity score used in the paper followed the MQM metric of assessing the severity of errors in text. Again, while this metric has been based in many papers, the accumulative nature of the suggested severity score causes it to suffer from monotinicity, which could not accurately represent the changes happening in the newly corrupted sentece. Additionally, the metric is discrete and this is could lead loss of information when attributing severity to an error.<br>
We propose two changes to the severity score metric which will allow it to be non-monotonic and also continuous. For more details please refer to the `Research Proposal.pptx` file

### Results
![NLP poster with graph](https://user-images.githubusercontent.com/63671077/229360806-8e5369f6-04a8-4b94-a4be-c6a34733f649.jpg)


## How to run?
### Run new_xlm_mbart_data.py for English:
python3 new_xlm_mbart_data.py -num_var 10 -lang en_XX -src case_study_src -ref case_study_ref -save save_file_name -severity ['original','2_1','2_2'] -whole_words True

# Disclaimer
Most of the work done on this project was by the original authors of the paper! We simply added our methods and tested their effectiveness. All credits of the original work goes to the original authors. 

