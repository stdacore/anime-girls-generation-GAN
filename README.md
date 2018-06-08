# anime-girls-generation-GAN

> DCGAN, WGAN-GP, ACGAN, StarGAN

Implementation of GANs for different purposes.  

* DCGAN: randomly generate cute anime girls.
* WGAN-GP: randomly generate cute anime girls.
* ACGAN (based on SRResNet-DRAGAN): randomly generate cute anime girls with specific hair and eyes color.
* StarGAN (based on SRResNet-DRAGAN): change hair color (style transfer).

## Results
* ACGAN
<div>
    <img src='samples/main.jpg'>
</div>

* StarGAN
<div>
    <img src='samples/stargan_result.jpg'>
</div>

* DCGAN & WGAN-GP
<div>
    <img src='samples/dcgan_result.jpg'>
    <img src='samples/wgangp_result.jpg'>
</div>

## Training progress
* SRResNet-DRAGAN
<div>
    <img src='samples/sample_gan/demo.gif'>
</div>

* ACGAN  
<div>
    <img src='samples/sample_acgan/demo.gif'>
</div>

* StarGAN  
<div>
    <img src='samples/sample_stargan/demo.gif'>
</div>

## Usage
```
$python stargan_training.py --cuda --train_dir <training_data_dir> --tag_file <tag_file> --output_dir <output_dir>
```
```
$python stargan_testing.py --cuda --model_dir <model> --input_dir <images_dir>
```

## Dataset
30k from [MakeGirlsMoe](https://make.girls.moe)  
10k from [Konachan](http://konachan.net/)  
each picture has one hair color tag and one eyes color tag.  

preprocessing: 
* keep one face in one picture and crop out other things.  
* resize to 64\*64, which means the model should input with 64\*64 and have the same size outputs.  

tag format:
```
1,<Color> hair <Color> eyes 
2,<Color> hair <Color> eyes
3,<Color> hair <Color> eyes
4,<Color> hair <Color> eyes
.
.
.
```

## Realease Note
* 2018.06.09 release StarGAN

## Reference
* https://github.com/yunjey/StarGAN 
* https://github.com/bhpfelix/PyTorch-Face-Generator-SRResNet-based-ACGAN-DRAGAN
* https://arxiv.org/pdf/1708.05509.pdf

