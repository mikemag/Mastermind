# Results








There is overhead to using the GPU, of course, and there are tuning points for when you use the GPU vs. falling back to
the CPU. But in general even for reasonable games like 4p6c the GPU is quite close. And you can see that for 4p7c the
GPU is already pulling ahead on some algorithms. Because of this I really haven't played with tuning the cutoff much at all.

#### 4p6c

|Strategy|Initial Guess|Max Turns|GPU Mode|Average Turns|Time (s)|CPU Scores|GPU Scores|GPU Kernels
|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|:---:|
|First One|3456|7|CPU|4.6211|0.0008| 55,417 | | |
|Knuth|1122|5|CPU|4.4761|0.0085| 3,237,885 | | |   
| | | |Both|4.4761|0.0142| 62,675 | 3,373,488 | 100| 
|Most Parts|1123|6|CPU|4.3735|0.0082| 3,289,320 | | |   
| | | |Both|4.3735|0.0168| 61,113 | 3,412,368 | 109| 
|Entropy|1234|6|CPU|4.4159|0.0164| 3,320,344 | | |   
| | | |Both|4.4151|0.0168| 58,489 | 3,443,472 | 106| 
|Expected Size|1123|6|CPU|4.3935|0.0163| 3,256,505 | | |   
| | | |Both|4.3951|0.0169| 62,062 | 3,346,272 | 106| 

#### 4p7c

|Strategy|Initial Guess|Max Turns|GPU Mode|Average Turns|Time (s)|CPU Scores|GPU Scores|GPU Kernels
|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|:---:|
|First One|4567|8|CPU|5.0675|0.0017| 114,469 | | |
|Knuth|1234|6|CPU|4.8367|0.0345| 13,577,917 | | |   
| | | |Both|4.8367|0.0351| 116,345 | 14,478,030 | 267| 
|Most Parts|1123|6|CPU|4.7430|0.0326| 13,470,924 | | |   
| | | |Both|4.7430|0.0334| 121,453 | 14,245,133 | 257| 
|Entropy|1234|6|CPU|4.7397|0.0573| 13,183,270 | | |   
| | | |Both|4.7401|0.0337| 118,016 | 13,983,424 | 237| 
|Expected Size|1234|6|CPU|4.7530|0.0459| 13,164,912 | | |   
| | | |Both|4.7505|0.0331| 116,820 | 13,875,379 | 237| 

