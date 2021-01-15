"""
~~Last editted by Faisal As'ad, 01/10/2021~~

This script trains a plane stress constiutive law relating the second Piola-Kirchoff in-plane stresses
and the Green-Lagrange in-plane strains,  in the form of 

[S11;S22;S12] = C*[E11;E22,2E12] + NN([E11;E22;2E12])

where C is a symmetric 3x3 matrix withc coefficient c_ij, obtained by a linear fit of the training data.

The script reads in training/test data in the form of ([S11,S22,S12],[E11,E22,E12]) pairs, obtains a linear fit through the data,
and applies a neural network correction to account for nonlinearity. It then writes the NN to disk, and prints necessary
that must be provided to AERO-S to exploit the TorchMat Material Law (c_ij, for j>=i).


For any inquiries, contact Faisal As'ad at faisal3 (at) stanford (dot) edu

"""



####################################################################################
################################ USER INPUTS #######################################
####################################################################################

#### DATA PARAMETERS ####
Training_dir = "Training_data/"    #Directory including stress, strain training data
Test_dir = "Test_data/"            #Directory including stress, strain test data
Augment = False                    #Whether or not to augment the data to enforce symmetry of shear under negation
dim = 3                            #Dimension of data. 3 corresponds to plane stress/strain 
                                   #situation, and is the only option currently.

#### LINEAR FIT ####
LinearFitType = "Orth"              #Options: Sym, Orth, Iso, None. See LinearReg function for more details.

#### NN PARAMETERS ####
fc_struct = [dim, 20, 20, dim]     #Structure of fully connected dense layers in NN.
ActivationFunction = "tanh"        #Activation function. Currently support tanh and relu.

####  TRAINING PARAMETERS ####
NormalizeOutput = True             #Normalize output (stress) to have 0 mean and unit std during training. The unnormalization is handled automatically.
NormalizeInput = True              #Normalize input (strain) to have 0 mean and unit std during training.
RegFactor = 1.0e-5                 #L2 regularization factor, to prevent overtraining.
LearningRate = 0.8                 #LBFGS optimizer learning rate. Recommended to be close to 1. 
NumEpochs = 150                    #Number of training epochs. 

#### FOR AERO-S EXPLOITATION ####
ModelName = "NN_{}".format(ActivationFunction)

####################################################################################
################################ END USER INPUTS ###################################
####################################################################################







#Import necessary modules
import torch
import numpy
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)


def ReadData(SYM = False, DIR = "Training_data/"):
    """ This function reads ([E11,E22,2E12],[S11,S22,S12]) from disk, and also augments the data if required """

    dummy = numpy.loadtxt(DIR + "macro.strainxx.1", usecols = (0,1))
    ntp = dummy.shape[0]


    sym_n = 0
    if SYM:
        sym_n = 1

    xx = numpy.zeros((ntp * (sym_n + 1), dim))
    yy = numpy.zeros((ntp * (sym_n + 1), dim))

    xx_ori = numpy.zeros((ntp , dim))
    yy_ori = numpy.zeros((ntp , dim))

    xx_ori[:, 0] = numpy.loadtxt(DIR + "macro.strainxx.1", usecols=(1))
    xx_ori[:, 1] = numpy.loadtxt(DIR + "macro.strainyy.1", usecols=(1))
    xx_ori[:, 2] = numpy.loadtxt(DIR + "macro.strainxy.1", usecols=(1))

    yy_ori[:, 0] = numpy.loadtxt(DIR + "macro.stressxx.1", usecols=(1))
    yy_ori[:, 1] = numpy.loadtxt(DIR + "macro.stressyy.1", usecols=(1))
    yy_ori[:, 2] = numpy.loadtxt(DIR + "macro.stressxy.1", usecols=(1))

    xx[0:ntp, :], yy[0:ntp, :] = xx_ori, yy_ori



    if SYM:

        min_E = min(0,numpy.min(xx[:,2]))
        safety = numpy.greater(xx_ori[:,2],-min_E)

        i = 1
        xx[i * (ntp):(i + 1) * (ntp), 0] = xx_ori[:, 0] * safety
        xx[i * (ntp):(i + 1) * (ntp), 1] = xx_ori[:, 1] * safety
        xx[i * (ntp):(i + 1) * (ntp), 2] = -xx_ori[:, 2] * safety

        yy[i * (ntp):(i + 1) * (ntp), 0] = yy_ori[:, 0] * safety
        yy[i * (ntp):(i + 1) * (ntp), 1] = yy_ori[:, 1] * safety
        yy[i * (ntp):(i + 1) * (ntp), 2] = -yy_ori[:, 2] * safety

        xx = xx[~numpy.all(xx == 0, axis=1)]
        yy = yy[~numpy.all(yy == 0, axis=1)]

    return xx, yy


def LinearReg(xx, yy, type):

    """ 
    This performs a least squares linear fit through the data to obtain H such that ||yy- H xx||_F is minimized 

    The three supported modes are, in order of most least restrictive to most restrictive are:
    1) Sym - This only enforces that H is symmetric, but places no other restriction
    2) Orth - Orthotropic material, requires that axial and shear components are decoupled
    3) Iso - Isotropic material, x and y direction are indistinguishable

    If type=None, then no linear fit is used.
    """

    ntp = xx.shape[0]
    assert(dim == 3)

    if type == "Sym":
        # H = 
        # h0 h1 h3
        # h1 h2 h4
        # h3 h4 h5
        #
        # minimize X h - Y
        X = numpy.zeros((dim*ntp, 6), dtype=float)
        for i in range(ntp):
            X[3 * i,     :] = xx[i,0],xx[i,1],0.0,    xx[i,2], 0.0,    0.0
            X[3 * i + 1, :] = 0.0,    xx[i,0],xx[i,1],0.0,    xx[i,2], 0.0
            X[3 * i + 2, :] = 0.0,    0.0,    0.0,    xx[i,0],xx[i,1], xx[i,2]
        Y = numpy.reshape(yy, (-1))

        h = numpy.linalg.lstsq(X, Y, rcond=None)[0]

        H = numpy.array([[h[0], h[1], h[3]],[h[1], h[2], h[4]], [h[3], h[4], h[5]]])

    elif type == "Orth":
        # H =
        # h0 h1 0
        # h1 h2 0
        # 0  0 h3
        #
        # minimize X h - Y
        X = numpy.zeros((dim * ntp, 4), dtype=float)
        for i in range(ntp):
            X[3 * i, :]     = xx[i, 0], xx[i, 1], 0.0,        0.0
            X[3 * i + 1, :] = 0.0,      xx[i, 0], xx[i, 1],   0.0
            X[3 * i + 2, :] = 0.0,      0.0,      0.0,        xx[i, 2]
        Y = numpy.reshape(yy, (-1))

        h = numpy.linalg.lstsq(X, Y, rcond=None)[0]

        H = numpy.array([[h[0], h[1], 0.0], [h[1], h[2], 0.0], [0.0, 0.0, h[3]]])
    elif type == "Iso":
        # H =
        # h0 h1 0
        # h1 h0 0
        # 0  0 h2
        #
        # minimize X h - Y
        X = numpy.zeros((dim * ntp, 3), dtype=float)
        for i in range(ntp):
            X[3 * i, :]     = xx[i, 0], xx[i, 1], 0.0,      
            X[3 * i + 1, :] = xx[i, 1], xx[i, 0], 0.0
            X[3 * i + 2, :] = 0.0,      0.0,      xx[i, 2]
        Y = numpy.reshape(yy, (-1))

        h = numpy.linalg.lstsq(X, Y, rcond=None)[0]

        H = numpy.array([[h[0], h[1], 0.0], [h[1], h[0], 0.0], [0.0, 0.0, h[2]]])

    elif type == "None":
        # H =
        # 0 0 0
        # 0 0 0
        # 0 0 0
        #
        # minimize X h - Y
        H = numpy.zeros((dim,dim))

    else:
        raise Exception("Error! Linear fit type {} is not supported".format(type))

    return H




class Net_Map(torch.nn.Module):
    """
    This class defines the structure of the neural network which will act as a correction to the constutive law.
    The inout/output are scaled appropriately by the means and standard deviations to promote quicker convergence.
    *Note: NNs train better when their inputs/outputs are close to 0 since the activation functions
    exhibit their nonlinearity there.

    Predicted stress =  H*strain + NN(strain)

    """   
    def __init__(self, mu_xx, sig_xx, mu_yy, sig_yy):
        super(Net_Map, self).__init__()

        self.mu_xx = torch.tensor(mu_xx)
        self.sig_xx = torch.tensor(sig_xx)

        self.mu_yy = torch.tensor(mu_yy)
        self.sig_yy = torch.tensor(sig_yy)

        self.fc=torch.nn.ModuleList()

        for i in range(len(fc_struct)-1):
            self.fc.append(torch.nn.Linear(fc_struct[i], fc_struct[i+1]).double())
        
        if ActivationFunction == "tanh":
            self.af = torch.tanh
        elif ActivationFunction == "relu":
            self.af = torch.relu

    def forward(self, x_in):

        x = x_in
        x = (x-self.mu_xx)/self.sig_xx

        for hl in self.fc[0:-1]:
            x=self.af(hl(x))
        x = self.fc[-1](x)

        x = x*self.sig_yy+self.mu_yy
        return x

def NN_Train():

    """
    This function trains the neural network defined in the Net_Map() class.
    """

    torch.autograd.set_detect_anomaly(True)

       
  
    #Read data
    xx, yy = ReadData(Augment, Training_dir)
    xx_test, yy_test = ReadData(Augment, Test_dir)

    ntp = xx.shape[0]
    ntp_test = xx_test.shape[0]
    
    #Linear fit to get H
    H = LinearReg(xx, yy, LinearFitType)
    yy_linear_fit = numpy.dot(xx, H)
    yy_test_linear_fit = numpy.dot(xx_test, H)

    #This is the 'stress' that the NN training will see.
    yy_nn = yy-yy_linear_fit

    if NormalizeInput:
        sig_xx = numpy.std(xx, axis=0)
        mu_xx = numpy.mean(xx, axis=0)
    else: 
        sig_xx = numpy.ones((1, dim))
        mu_xx = numpy.zeros((1, dim))

    if NormalizeOutput:
        sig_yy = numpy.std(yy_nn, axis=0)
        mu_yy = numpy.mean(yy_nn, axis=0)
    else: 
        sig_yy = numpy.ones((1, dim))
        mu_yy = numpy.zeros((1, dim))

    model = Net_Map(mu_xx, sig_xx, mu_yy, sig_yy)

    inputs = torch.from_numpy(xx).view(ntp, dim)
    outputs = torch.from_numpy(yy_nn).view(ntp, dim)

    optimizer = optim.LBFGS(model.parameters(), lr=LearningRate, max_iter=10, line_search_fn='strong_wolfe')

    #Scaled regularization factor
    factor = torch.tensor(RegFactor* ntp)

    #Used for normalizing the printed loss.
    init_loss1 = torch.sum( (outputs/sig_yy) ** 2 )

    #Training loop
    for i in range(NumEpochs):
        def closure():
            """ 
            This function defines the loss function for the NN training. 
            The two constributions are 1) the squared error in the stress data and predicted stress
            and 2) an L2 regularization term to prevent overfitting.
            """

            optimizer.zero_grad()

            #Evaluate model
            sigma = model(inputs)

            #Accumilate L2 regularization penalty
            l2_loss = torch.tensor(0.)
            for param in model.parameters():
                l2_loss = l2_loss.add( param.norm() ** 2 )

            #Function fitting loss contribution. THe contributions are renormalized here by sig_yy since the forward pass unnormalizes them.
            loss1 = torch.sum(((sigma - outputs) / torch.tensor(sig_yy) ) ** 2 )

            #L2 penalty loss contribution
            loss2 = factor * l2_loss

            #Total loss
            loss = loss1 + loss2
           

            loss.backward(retain_graph=True)
            print("Epoch {}, relative NN error = {}".format(i, (loss1 / init_loss1) ** 0.5))
            return loss
        optimizer.step(closure)



    #Evaluate trained model on training data set
    yy_pred = model(torch.from_numpy(xx).view(ntp, dim))
    yy_pred = yy_pred.data.numpy()
    res_train = yy - yy_linear_fit - yy_pred

    print('\n*RELATIVE TRAINING SET ERRORS')
    print("S_11 : {:.2f}%, S_22 : {:.2f}%, S_12 : {:.2f}%".format(
          numpy.linalg.norm(res_train[:,0:1], ord='fro') / numpy.linalg.norm(yy[:,0:1], ord = 'fro')*100,
          numpy.linalg.norm(res_train[:,1:2], ord='fro') / numpy.linalg.norm(yy[:,1:2], ord = 'fro')*100,
          numpy.linalg.norm(res_train[:,2:3], ord='fro') / numpy.linalg.norm(yy[:,2:3], ord = 'fro')*100))


    #Evaluate trained model on test data set
    yy_test_pred = model(torch.from_numpy(xx_test).view(ntp_test, dim))
    yy_test_pred = yy_test_pred.data.numpy()
    res_test = yy_test - yy_test_linear_fit - yy_test_pred

    print("\n*RELATIVE TEST SET ERRORS")
    print("S_11 = {:.2f}%, S_22 : {:.2f}%, S_12 : {:.2f}%".format(
          numpy.linalg.norm(res_test[:,0:1], ord='fro') / numpy.linalg.norm(yy_test[:,0:1], ord = 'fro')*100,
          numpy.linalg.norm(res_test[:,1:2], ord='fro') / numpy.linalg.norm(yy_test[:,1:2], ord = 'fro')*100,
          numpy.linalg.norm(res_test[:,2:3], ord='fro') / numpy.linalg.norm(yy_test[:,2:3], ord = 'fro')*100))


    #Save trained model to cpp file
    example = torch.rand([1, dim]).double()
    traced_script_module = torch.jit.trace(model, example)
    output = traced_script_module(torch.ones([1, dim]).double())
    save_name ="model_{}.pt".format(ModelName)
    traced_script_module.save(save_name)
    print("\n*Model saved to disk as \'{}\'".format(save_name))

    #Print relevant information for utilization of the TorchMat material law in AERO-S.

    print("\n*To successfully exploit the saved model for the TorchMat Material Law in AERO-S, you must provide the following parameters:\n")
    for i in range(dim):
        for j in range(i,dim):
            print(' c{}{} = {}'.format(i,j,H[i,j]))


if __name__ == "__main__":

    NN_Train()




