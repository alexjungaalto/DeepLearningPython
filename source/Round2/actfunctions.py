def actfunctions():
    
    '''
    Function for plotting activation functions and its gradients.
    '''
    
    import numpy as np
    import matplotlib.image as mpimg
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from ipywidgets import interactive, IntSlider, Layout, HBox, VBox
    plt.style.use('seaborn')
    
    mode =  input("Enter mode. Valid names are:\n'actfunc', 'gradient'\n")
    actfunc = input("Enter the name of the activation function. Valid names are:\n'relu',"\
                    "'leakyrelu', 'elu', sigmoid', 'tanh'\n")
   
 #------DEFINE ACTIVATION FUNCTIONS AND ITS DERIVATIVES----------------------#
    def relu(z):
            g = np.copy(z)
            g[z<0] = 0
            g_grad = np.ones(shape=g.shape)
            g_grad[z<0] = 0
            return g, g_grad
    
    def leakyrelu(z):
            alpha = 0.1
            g = np.copy(z)
            g[z<0] = alpha*z[z<0]
            g_grad = np.ones(shape=g.shape)
            g_grad[z<0] = alpha
            return g, g_grad

    def elu(z):
            alpha = 1
            g = np.copy(z)
            g[z<0] = alpha*(np.exp(z[z<0])-1)
            g_grad = np.ones(shape=g.shape)
            g_grad[z<0] = g[z<0] + alpha
            return g, g_grad

    def sigmoid(z):
            g = 1/(1+np.exp(-z))
            g_grad = g*(1-g)
            return g, g_grad
        
    def tanh(z):
            g = 2/(1+np.exp(-2*z)) - 1
            g_grad = 1-g**2
            return g, g_grad
 #-------------------------------------------------------------------------#

    # dictionary for calling the act.function by its name
    actfunc_dict = {'relu':relu, 'leakyrelu':leakyrelu,'elu':elu, 'sigmoid':sigmoid, 'tanh':tanh}
    # select act.function 
    act = actfunc_dict[actfunc]
    
 #------PRINT INTRO INFORMATION ON ACT.FUNCTIONS AND PLOT ITS GRAPH-------------#
    def background(actfunc):
        
        # some basic info about activation function
        if actfunc=='relu':
            txt = "Rectifying Linear Unit (ReLU) non-linearity has the mathematical form  $max(0,z)$."\
                  "\nReLU simply performs thresholding the input at zero."\
                  "\nReLU computationally cheaper than, e.g. sigmoid, as it does not require the computaion of exponent."\
                  "\nUse of ReLU in the NN seems to result in faster SGD convergance, compare to sigmoid or tanh."\
                  "\nReLU neurons are prone to 'dying' - the process when the neuron output and gradient is always zero."\
                  "\nThe possible reason for the 'dying' ReLU neurons is large gradient flow, which can lead to large negative weight or bias."
        
        elif actfunc=='leakyrelu':
            txt = "Leaky ReLU non-linearity has the mathematical form:"\
                  "\n$g(z) = z, for \;z\geq0$"\
                  "\n$g(z)= {\\alpha}z, for \;z<0$"\
                  "\nLeaky ReLU tries to fixed the problem of 'dying' ReLU neurons by multiplying the input $z<0$ with value $\\alpha$."

        elif actfunc=='elu':
            txt = "Exponential Linear Unit (ELU) non-linearity has the mathematical form:"\
                  "\n$g(z)=z, for \;z\geq0$"\
                  "\n$g(z)= {\\alpha}(e^{\\bfz}-1), for \;z<0$"\
                  "\nCompare to ReLU and leaky ReLU, ELU is smooth and differentiable function."\
                  "\nLike leaky ReLU, ELU neurons are not prone to 'dying'."
        
        elif actfunc=='sigmoid':
            txt = "The sigmoid non-linearity has the mathematical form  $\\frac{1}{1+exp^{\\bf-z}}$."\
                   "\nTakes a real-valued number and “squashes” it into range between 0 and 1."\
                   "\nSigmoids saturate and 'kill' gradients, as the gradient at the flat regions of the sigmoid curve is nearly zero."\
                   "\nSigmoid outputs are not zero-centered, which leads to zig-zagging of gradient updates."
        
        elif actfunc=='tanh':
            txt = "The tanh non-linearity has the mathematical form  $\\frac{2}{1+e^{\\bf-2z}}$."\
                  "\nSimilar to sigmoid, tanh prone to saturation and 'killing' the gradient."\
                  "\nUnlike sigmoid, tanh “squashes” input into range between -1 and 1,"\
                  "\nthus output of tanh is zero-centered  and not prone to zig-zagging of gradient updates."
              
        # plot the activation function
        # create the input
        x = np.linspace(-10, 10, 100)
        # apply activation function to the input
        y = act(x)[0]
        
        fig, axes = plt.subplots(1,1,figsize=(2,2))    
        axes.plot(x,y)
        # display the info about activation function near plot
        axes.text(1.2, 0, txt, fontsize=18, transform=axes.transAxes)
        plt.ylim(y.min()-0.5, y.max()+0.5)
        plt.show()
     
    # compute and plot activations of the linear outputs of two neurons
    def activation(w1, w2, w3, w4, b1, b2, b3, actfunc):
        
        #-----DEFINE LINEAR AND NON-LINEAR MODELS--------#
        x = np.linspace(-10, 10, 100)
        # create linear models for 2 neurons
        y1 = w1*x + b1
        y2 = w2*x + b2
        # apply activation function to the linear functions
        g1 = act(y1)[0]
        g2 = act(y2)[0]
        # scale and shift the result of activation
        g12 = w3*g1 
        g22 = w4*g2 
        # sum up the activations of two neurons
        a = g12 + g22 + b3
        
        #-----PLOT ALL FUNCTIONS--------#
        plt.style.use('seaborn')
        
        fig = plt.figure(figsize=(15,6))
        gs = GridSpec(2, 5, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])

        ax1.plot(x, y1)
        ax2.plot(x, y2)
        ax1.set_title('neuron #1 \n$z_1 = w_1x+b_1$', fontsize=16)
        ax2.set_title('neuron #2 \n$z_2 = w_2x+b_2$', fontsize=16)

        ax3 = fig.add_subplot(gs[0, 1])
        ax4 = fig.add_subplot(gs[1, 1])

        ax3.plot(x, g1)
        ax4.plot(x, g2)
        ax3.set_title('$g(z_1) = '+actfunc+'(z_1)$', fontsize=16)
        ax4.set_title('$g(z_2) = '+actfunc+'(z_2)$', fontsize=16)

        ax4 = fig.add_subplot(gs[0, 2])
        ax5 = fig.add_subplot(gs[1, 2])

        ax4.plot(x, g12)
        ax5.plot(x, g22)
        ax4.set_title('$w_3g(z_1)$', fontsize=16)
        ax5.set_title('$w_4g(z_2)$', fontsize=16) 

        ax6 = fig.add_subplot(gs[:,3:])

        ax6.plot(x, a)
        ax6.set_title("weighted sum of hidden neurons' activations \n$w_3g(z_1)+w_4g(z_2)+b_3$", fontsize=16)

        fig.tight_layout()
        plt.show()

        return 
                    
                  
    # plot gradients
    def gradient(w, b, actfunc):

        x_grid = np.linspace(-10, 10, 1000)
        # create linear models for 1 neuron
        z = w*x_grid + b
        # apply activation function to the linear function
        g, g_grad = act(z)
        
        # plot all functions
        
        fig = plt.figure(figsize=(12,6))
        gs = GridSpec(2, 3, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        ax1.plot(x_grid, z)
        ax2.plot(x_grid, g)
        ax1.set_title('$z = wx+b$', fontsize=16)
        ax2.set_title('$g(z)$', fontsize=16)
        ax2.set_ylim(g.min()-0.25, g.max()+0.25)

        ax3 = fig.add_subplot(gs[1, 1])

        ax3.plot(x_grid, g_grad)
        ax3.set_title('local gradient $\\frac{\partial g(z)}{\partial z}$', fontsize=20)
        ax3.set_ylim(g_grad.min()-0.25, g_grad.max()+0.25)
        
        ax4 = fig.add_subplot(gs[1, 0])

        ax4.plot(x_grid, x_grid)
        ax4.set_title('local gradient  $\\frac{\partial z(w)}{\partial w}$', fontsize=20)
        
        ax5 = fig.add_subplot(gs[:, 2])

        ax5.plot(x_grid, x_grid*g_grad)
        ax5.set_title('$\\frac{\partial z(w)}{\partial w} \\frac{\partial g(z)}{\partial z}$', fontsize=20)

        fig.tight_layout()
        plt.show()

        return
    
    if mode=='actfunc':
        # call `background` function to print act.function info
        background(actfunc) 
                  
        # create interactive sliders to change values of weights and biases
        w1 = IntSlider(min=-10, max=10, description='$w_1$')
        w2 = IntSlider(min=-10, max=10, description='$w_2$')
        w3 = IntSlider(min=-10, max=10, description='$w_3$')
        w4 = IntSlider(min=-10, max=10, description='$w_4$')

        b1 = IntSlider(min=-10, max=10, description='$b_1$')
        b2 = IntSlider(min=-10, max=10, description='$b_2$')
        b3 = IntSlider(min=-10, max=10, description='$b_3$')

        # create widget 
        widget = interactive(activation, actfunc=actfunc, w1=w1, w2=w2, w3=w3, w4=w4,
                            b1=b1, b2=b2, b3=b3)
        # order the grouping of the sliders
        controls = VBox([HBox([w1, b1]), HBox([w2, b2]), HBox([w3, b3]), HBox([w4])])
        output = widget.children[-1]
        display(controls, output)
    
    elif mode=='gradient':             
    
        # create interactive sliders to change values of weights and biases
        w = IntSlider(min=-10, max=10, description='$weight \;w$')
        b = IntSlider(min=-10, max=10, description='$bias \;b$')

        # create widget 
        widget = interactive(gradient, actfunc=actfunc, w=w, b=b)
        # order the grouping of the sliders
        controls = HBox([w, b])
        output = widget.children[-1]
        display(controls, output)
    
    return