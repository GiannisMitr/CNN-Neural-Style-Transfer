# CNN-Neural-Style-Transfer

A convolutional neural network performing style transfer between two images. Implemented with tensorflow

Uses a VGG-19 network pretrained on ImageNet Dataset. Middle layers activations of content image 
and style image are utilized to define a composite cost. Then the generated image is run through the network and
the defined cost is optimized with Adam optimization algorithm with respect to the generated image input. 
That way the generated image changes from random pixels to a composition of the content and style images.   

### Example generation

<p float="left">
  <img src="/images/content.jpg" alt="drawing" width="150"/>
  <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b>
  <img src="/images/style.jpg" alt="drawing" width="150"/>
  <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b>
 <img src="/images/generated.png" alt="drawing" width="150"/>
</p>  

**&nbsp;&nbsp;&nbsp;&nbsp;content image (C) &nbsp;&nbsp;&nbsp;  + &nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;style image (S) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;=  &nbsp;&nbsp;  &nbsp;generated image (G)** 
 <br><br/>
### Cost definition

The total cost is computed as follows

<img src="https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Cbg_white%20%5Clarge%20%24%24J%28G%29%20%3D%20%5Calpha%20J_%7Bcontent%7D%28C%2CG%29%20&plus;%20%5Cbeta%20J_%7Bstyle%7D%28S%2CG%29%24%24" align="middle"/> 

where

* **J<sub>content</sub>(C,G)** : The **L<sub>2</sub>** norm squared of the diffence of activations of content image(C) and generated image (G) in a particular layer of the network. 
* **J<sub>content</sub>(S,G)** : The **Frobenius** norm of the diffence of the Gramian matrices of activations for style image(S) and generated image (G), computed and averaged across five layers of the network. 
* **α,β** : the relative weighing of the two costs to the total cost.
 <br><br/>
## Usage

Execute                      *style_transfer* python script. By configuring *content_image* and *style_image* paths you can transfer style between your images. The input images must be 400x300 pixels (width/height). You can change the learning_rate, and the layers used for defining the cost **J(C)** and **J(S)**, also you can tweak their relative weighing on the total cost.


The code will:
* Train the defined model for configurable number of iterations
* Every 100 iterations will store the generated image to the */output* path.


*Based on code and lectures from the Deeplearning.ai specialization.*
 <br><br/>
## References

Deeplearning.ai CNN course on Coursera.[[1]](https://www.coursera.org/learn/convolutional-neural-networks)   
Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). A Neural Algorithm of Artistic Style.[[2]](https://arxiv.org/abs/1508.06576)    
Karen Simonyan and Andrew Zisserman (2015). Very deep convolutional networks for large-scale image recognition.[[3]](https://arxiv.org/pdf/1409.1556.pdf)  
MatConvNet. Pretrained VGG-19 network on imageNet.[[4]](http://www.vlfeat.org/matconvnet/pretrained/)