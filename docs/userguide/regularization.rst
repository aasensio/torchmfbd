.. include:: ../code_name
.. _configuration:


Regularization
==============

|torchmfbd| allows the user to define regularization terms to be added to the loss function by inherting from ``torchmfbd.Regularization``. The smooth and IUWT regularization terms
can be added via the configuration file. We will add more options in the future but you can define your own regularization 
terms via an external function.

As an example, let us assume that we want to add a new regularization term to the loss function that penalizes 
values of the object away from zero. To this end, we define the following class:

::

    class MyRegularization(torchmfbd.Regularization):
        def __init__(self, lambda_reg, variable, value):
            super(MyRegularization, self).__init__('external', lambda_reg, variable)

            self.variable = variable
            self.lambda_reg = lambda_reg
            self.value = value            

        def __call__(self, x):

            # Add your regularization term here
            n_o = len(x)
            loss = 0.0
            for i in range(n_o):
                loss += self.lambda_reg * torch.sum((x[i] - self.value)**2)

            return loss

Now we instantiate the ``torchmfbd.Deconvolution`` class and add the external regularization:
    
::

    deconv = torchmfbd.Deconvolution('qs_8542_kl_patches.yaml')
    
    myregularization = MyRegularization(lambda_reg=0.01, variable='object', value=0.0)
    deconv.add_external_regularization(myregularization)

The external regularization term will be added to the loss function and will be optimized along with the other terms.