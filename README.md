# PyTorch-HITNet-Hierarchical-Iterative-Tile-Refinement-Network-for-Real-time-Stereo-Matching
HITNet implementation using PyTorch

This is a repository including code implementing Google paper ***HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching***

This project is an initial version, which can train and test the model but may contain some errors and need further modification and debugging. If you find any issue about my code, please open issues or contact me (*mjitglv@gmail.com*) as soon as possible. 

Currently this project cannot reproduce the accuracy and speed reported in the original paper. In terms of the speed, the official implementation uses their optimized cuda op to accelerate the reference and training.(Please refer to their [official repository](https://github.com/googleresearch/google-research/tree/master/hitnet), which has not inculded the model code yet). 

Thanks for the help of Vladimir Tankovich, who has proposed this great stereo network with his team and provided me with a lot of details and clarifications of the original paper.

Also, I would thank @xy-guo, who proposed the amazing [GwcNet](https://github.com/xy-guo/GwcNet), since the code was partially borrowed from his repository.
