# A Convolutional Neural Network performing style transfer between two images.
# Uses a VGG-19 network with pretrained weights on ImageNet.
# Implemented with TensorFlow. 
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from utils import *
import numpy as np
import tensorflow as tf
import time


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_C_unrolled = tf.reshape(a_C,[-1])
    a_G_unrolled = tf.reshape(a_G,[-1])
    
    J_content = tf.multiply(1/(4*n_H*n_W*n_C),tf.square(tf.norm(tf.subtract(a_C_unrolled,a_G_unrolled))))

    return J_content



def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    GA = tf.matmul(A,tf.transpose(A))
    
    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
 
    a_S = tf.reshape(tf.transpose(a_S),[n_C,n_H*n_W])
    a_G = tf.reshape(tf.transpose(a_G),[n_C,n_H*n_W])

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    
    J_style_layer = tf.square(tf.norm(tf.subtract(GS,GG),ord='fro', axis=(0,1)))/(4*(n_H*n_W)**2*(n_C**2))
    
    return J_style_layer



STYLE_LAYERS = [
    ('conv1_1', 0.1),
    ('conv2_1', 0.15),
    ('conv3_1', 0.3),
    ('conv4_1', 0.15),
    ('conv5_1', 0.3)]



def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    
    J = alpha*J_content + beta * J_style

    return J


# 1. Create an Interactive Session
# 2. Load the content image 
# 3. Load the style image
# 4. Randomly initialize the image to be generated 
# 5. Load the VGG16 model
# 7. Build the TensorFlow graph:
#     - Run the content image through the VGG16 model and compute the content cost
#     - Run the style image through the VGG16 model and compute the style cost
#     - Compute the total cost
#     - Define the optimizer and the learning rate
# 8. Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.


tf.reset_default_graph()
sess = tf.InteractiveSession()


# Load, reshape, and normalize "content" image:
content_image = scipy.misc.imread("images/pic_small.jpg")
content_image = reshape_and_normalize_image(content_image)


# Load, reshape, and normalize "style" image:
style_image = scipy.misc.imread("images/picasso.jpg")
style_image = reshape_and_normalize_image(style_image)


# Initialize the "generated" image as a noisy image created from the content_image.
# By initializing the pixels of the generated image to be mostly noise but still slightly correlated with the content image,
# this will help the content of the "generated" image more rapidly match the content of the "content" image.
generated_image = generate_noise_image(content_image)

# Load the VGG16 model.
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

# Assign the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# and isn't evaluated yet. Later in the code, when we run the session, this will be the activations drawn
# from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)

# Compute the total cost
J = total_cost(J_content, J_style, 10, 40)

# define Adam optimizer with learning rate
optimizer = tf.train.AdamOptimizer(4.0)

# define train_step
train_step = optimizer.minimize(J)

def model_train(sess, input_image, num_iterations = 10000):
    
    # Initialize global variables
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model.
    sess.run(model['input'].assign(input_image))
    oldTime = time.time()
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        _ = sess.run(train_step)
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])
        # Print every 20 iteration.
        if i%100 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            print("last iterations took: " +str(time.time() - oldTime))
            oldTime = time.time()
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
            
    # save current generated image in the "/output" directory
    save_image('output/final_image.jpg', generated_image)
    return generated_image


model_train(sess, generated_image)
