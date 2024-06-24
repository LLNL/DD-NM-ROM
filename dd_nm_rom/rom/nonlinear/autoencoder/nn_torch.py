# autoencoder.py

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sparselinear import SparseLinear
import torch.nn.utils.prune as prune
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from time import time
import sys, copy
import matplotlib.pyplot as plt
from dd_nm_rom import backend as bkd
from dd_nm_rom.ops import map_nested_dict



def generate_mask(output_dim, row_nnz, row_shift, print_sparsity=False):
  '''
  Generates a sparsity mask for decoder.

  inputs:
  output_dim: dimension of the decoder output
  row_nnz:    number of nonzero elements per row of mask
  row_shift:  amount to shift nonzero band per row
  print_sparsity: [optional] Boolean to print percent sparsity of mask and produces spy plot. Default is False

  outputs:
  mask: (output_dim, latent_dim) sparse matrix of sparsity mask
  hidden_dim: dimension of hidden layer. Depends on row_nnz and shift
  '''
  hidden_dim = int(row_nnz + row_shift*(output_dim-1))

  e = np.ones(row_nnz, dtype='int8')
  ind = np.arange(row_nnz)
  row = np.array([], dtype='int8')
  col = np.array([], dtype='int8')

  for i in range(output_dim):
    row = np.append(row, i*e)
    col = np.append(col, ind + i*row_shift)

    if (row_shift+1)*row_nnz + i*row_shift - 1 < hidden_dim:
      row = np.append(row, i*e)
      col = np.append(col, ind + i*row_shift + row_shift*row_nnz )

    if  -row_shift*row_nnz + i*row_shift >= 0:
      row = np.append(row, i*e)
      col = np.append(col, ind + i*row_shift - row_shift*row_nnz)

  data = np.ones(row.size, dtype='int8')
  mask = sp.coo_matrix((data, (row, col)), shape=(output_dim, hidden_dim))

  if print_sparsity:
    print(f"Sparsity in {mask.shape} mask: {(1.0-mask.nnz/np.prod(mask.shape))*100:.2f}%")
    plt.figure(figsize=(12,8))
    plt.spy(mask)
    plt.show()

  return mask, hidden_dim

# swish activation function
class Swish(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x*torch.sigmoid(x)

class Linear(nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x

def RelMSELoss(outputs, pred):
  '''
  Relative MSE Loss: loss = (1/N) (Sum_{i=1}^N || output[i] - pred[i] ||^2 / || output[i] ||^2)

  inputs:
  outputs: (N, output_dim) tensor of given output data
  pred:    (N, output_dim) tensor of predicted output data

  outputs:
  loss:    Relative MSE loss
  '''
  return torch.mean(torch.sum(torch.square(outputs-pred), dim=1)/(torch.sum(torch.square(outputs), dim=1)+1e-6))


class Identity(torch.nn.Module):
  '''
  Generic class for decoder part of autoencoder.
  The struture is shallow with one hidden layer.

  inputs:
  latent_dim: dimension of latent dimension
  hidden_dim: dimension of linear hidden layer
  output_dim: dimension of outputs
  mask:       sparsity mask in coo format
  scale:      (output_dim) tensor for scaling input data
  ref:        (output_dim) tensor for shifting input data
  activation: [optional] activation function between hidden and output layer. 'Swish' or 'Sigmoid'. Default is 'Sigmoid'
  '''
  def __init__(self, dim, mask, scale, ref):
    super(Identity, self).__init__()
    self.dim = dim
    self.ref = ref
    self.scale = scale
    lay1 = SparseLinear(dim, dim, connectivity=torch.LongTensor(np.vstack((mask.row, mask.col))))
    lay2 = SparseLinear(dim, dim, connectivity=torch.LongTensor(np.vstack((mask.row, mask.col))), bias=False)
    self.net = torch.nn.Sequential(lay1, Linear(), lay2)
    self.net.apply(self.init_weights)

  def init_weights(self, m):
    if isinstance(m, SparseLinear):
      torch.nn.init.ones_(m.weights)
      if m.bias is not None:
        torch.nn.init.zeros_(m.bias)

  def forward(self, x):
    return self.net(x)

# Encoder class
class Encoder(nn.Module):
  '''
  Generic class for encoder part of autoencoder.
  The struture is shallow with one hidden layer.

  inputs:
  input_dim:  dimension of input data
  hidden_dim: dimension of linear hidden layer
  latent_dim: dimension of latent dimension
  mask:       sparsity mask in coo format
  scale:      (input_dim) tensor for scaling input data
  ref:        (input_dim) tensor for shifting input data
  activation: [optional] activation function between hidden and output layer. 'Swish' or 'Sigmoid'. Default is 'Sigmoid'
  '''
  def __init__(self, input_dim, hidden_dim, latent_dim, mask,
         scale, ref,
         act_fun=nn.Sigmoid, min_dim=5):
    super(Encoder, self).__init__()
    self.scale = scale
    self.ref = ref
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim
    self.net = nn.Sequential(
      SparseLinear(input_dim, hidden_dim, connectivity=torch.LongTensor(np.vstack((mask.row, mask.col)))),
      act_fun(),
      nn.Linear(hidden_dim, latent_dim, bias=False),
    )
  '''
  Evaluate encoder.

  input:
  w: (input_dim) tensor of input data

  output:
  output: (latent_dim) tensor of output data
  '''
  def forward(self, w):
    return self.net((w-self.ref)/self.scale)

class Encoder_dense(nn.Module):
  '''
  Generic class for dense encoder part of autoencoder.
  The struture is shallow with one hidden layer.

  inputs:
  input_dim:  dimension of input data
  hidden_dim: dimension of linear hidden layer
  latent_dim: dimension of latent dimension
  scale:      (input_dim,) tensor for scaling input data
  ref:        (input_dim,) tensor for shifting input data
  activation: [optional] activation function between hidden and output layer. 'Swish' or 'Sigmoid'. Default is 'Sigmoid'
  '''
  def __init__(self, input_dim, hidden_dim, latent_dim,
         scale, ref,
         act_fun=nn.Sigmoid):
    super(Encoder_dense, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim
    self.scale = scale
    self.ref = ref
    self.net = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      act_fun(),
      nn.Linear(hidden_dim, latent_dim, bias=False),
    )

  '''
  Evaluate encoder.

  input:
  w: (input_dim) tensor of input data

  output:
  output: (latent_dim) tensor of output data
  '''
  def forward(self, w):
    return self.net((w-self.ref)/self.scale)

class Decoder(nn.Module):
  '''
  Generic class for decoder part of autoencoder.
  The struture is shallow with one hidden layer.

  inputs:
  latent_dim: dimension of latent dimension
  hidden_dim: dimension of linear hidden layer
  output_dim: dimension of outputs
  mask:       sparsity mask in coo format
  scale:      (output_dim) tensor for scaling input data
  ref:        (output_dim) tensor for shifting input data
  activation: [optional] activation function between hidden and output layer. 'Swish' or 'Sigmoid'. Default is 'Sigmoid'
  '''
  def __init__(self, latent_dim, hidden_dim, output_dim, mask,
         scale, ref,
         act_fun=nn.Sigmoid):
    super(Decoder, self).__init__()
    self.latent_dim = latent_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.scale = scale
    self.ref = ref
    self.net = nn.Sequential(
      nn.Linear(latent_dim, hidden_dim),
      act_fun(),
      SparseLinear(hidden_dim, output_dim, connectivity=torch.LongTensor(np.vstack((mask.row, mask.col))), bias=False),
    )

  '''
  Evaluate decoder.

  input:
  w: (latent_dim) tensor of input data

  output:
  output: (output_dim) tensor of output data
  '''
  def forward(self, w):
    return self.scale*self.net(w)+self.ref

class Autoencoder:
  '''
  Class for implementing and training an autoencoder with sparse-masked decoder for model reduction.

  inputs:
  snapshots:  (n_snapshots, input_dim) array of snapshot training data
  latent_dim: latent dimension of autoencoder
  row_nnz:    number of nonzeros per row of sparsity mask
  row_shift:  amount to shift nonzero band per row in sparsity mask
  device:     PyTorch device. 'cpu' or 'cuda'
  dense_encoder: [optional] Boolean for using dense encoder. Default is False.
  encoder_hidden: [optional] dimension of encoder hidden layer. Default uses 2*input_dim
  act_type:   [optional] activation function for encoder and decoder. 'Sigmoid' or 'Swish'. Default is 'Sigmoid'
  test_prop:  [optional] proportion of snapshots data to be used for testing set. Default is 0.1
  lr:         [optional] learning rate. Default is 1e-3
  lr_patience:[optional] patience for learning rate scheduler. Default is 10
  seed:       [optional] random seed. Default is None

  fields:
  device:      PyTorch device. 'cpu' or 'cuda'
  lr:          learning rate. Default is 1e-3
  lr_patience: patience for learning rate scheduler. Default is 10
  ref:         reference vector for normalizing snapshot data
  scale:       scaling vector for normalizing snapshot data
  n_snapshots: number of snapshot data
  test_size:   size of testing set
  train_size:  size of training set
  test_data:   TensorDataset of testing data
  train_data:  TensorDataset of training data
  input_dim:   dimension of snapshot data
  latent_dim:  latent space dimension
  output_dim:  output space dimension
  row_nnz:     number of nonzeros per row of sparsity mask
  row_shift:   amount to shift nonzero band in per row in sparsity mask
  mask:        sparsity mask for decoder output layer (transpose of mask for encoder input layer)
  encoder_hidden: dimension of encoder hidden layer
  decoder_hidden: dimension of decoder hidden layer
  encoder:     instance of Encoder class
  decoder:     instance of Decoder class
  optimizer:   optimizer of autoencoder. Adam is used
  scheduler:   learning rate scheduler. Reduce LR on Plateau is used
  loss_fn:     loss function. MSELoss is used

  methods:
  forward:  forward pass through autoencoder, i.e. decoder(encoder(data))
  train:    train autoencoder
  '''
  def __init__(self, snapshots, latent_dim, row_nnz, row_shift, device,
         min_dim=5,
         dense_encoder=False,
         encoder_hidden=-1,
         act_type='Sigmoid',
         test_prop=0.1,
         lr=1e-3,
         lr_patience=10,
         loss='AbsMSE',
         seed=None, filename='net_dict.p'):

    self.filename = filename

    self.trainable = True

    self.device = device
    self.lr = lr
    self.lr_patience = lr_patience

    # compute scaling vectors
    self.ref = 0.5*(snapshots.max(0)[0]+snapshots.min(0)[0]).to(self.device)
    self.scale = 0.5*(snapshots.max(0)[0]-snapshots.min(0)[0]).to(self.device)

    # data sizes
    self.n_snapshots  = snapshots.shape[0]
    self.test_size = int(test_prop*self.n_snapshots)
    self.train_size = self.n_snapshots - self.test_size

    # separate into test and train set
    rng = np.random.default_rng(seed)
    all_ind   = np.arange(self.n_snapshots)
    test_ind  = rng.choice(all_ind, size=self.test_size, replace=False)
    train_ind = np.setdiff1d(all_ind, test_ind)

    self.train_data = TensorDataset(snapshots[train_ind])
    self.test_data = TensorDataset(snapshots[test_ind])

    # Minimum dimension to use an autoencoder
    self.min_dim = min_dim

    # sizes of layers in encoder and decoder
    self.input_dim = snapshots.shape[1]
    self.output_dim = self.input_dim

    if self.input_dim <= self.min_dim:

      self.latent_dim = self.input_dim
      self.decoder_hidden = self.input_dim
      self.encoder_hidden = self.input_dim
      self.act_type = "Linear"
      self.mask = sp.eye(self.input_dim).tocoo()

      self.encoder = Identity(
        dim=self.input_dim,
        mask=self.mask.T,
        scale=self.scale,
        ref=self.ref
      ).to(self.device)

      self.decoder = Identity(
        dim=self.input_dim,
        mask=self.mask,
        scale=self.scale,
        ref=self.ref
      ).to(self.device)

      self.best_encoder = copy.deepcopy(dict(self.encoder.state_dict()))
      self.best_decoder = copy.deepcopy(dict(self.decoder.state_dict()))

      torch.save(self.state_dict_np(), self.filename[:-2]+"_numpy.p")

      self.trainable = False

    else:

      self.latent_dim = latent_dim

      # compute sparsity mask
      self.row_nnz = row_nnz
      self.row_shift = row_shift

      self.mask, self.decoder_hidden = generate_mask(self.output_dim,
                                                      self.row_nnz,
                                                      self.row_shift,
                                                      print_sparsity=False)

      self.act_type = act_type
      if act_type == 'Sigmoid':
          self.act_fun = nn.Sigmoid
      elif act_type == 'Swish':
          self.act_fun = Swish
      elif act_type == 'ELU':
          self.act_fun = nn.ELU

      # initialize encoder and decoder
      if dense_encoder:
          self.encoder_hidden = 2*self.input_dim if encoder_hidden < 0 else encoder_hidden
          self.encoder = Encoder_dense(self.input_dim,
                                        self.encoder_hidden,
                                        self.latent_dim,
                                        self.scale,
                                        self.ref,
                                        act_fun=self.act_fun).to(self.device)
      else:
          self.encoder_hidden = self.decoder_hidden
          self.encoder = Encoder(self.input_dim,
                                  self.encoder_hidden,
                                  self.latent_dim,
                                  self.mask.T,
                                  self.scale,
                                  self.ref,
                                  act_fun=self.act_fun).to(self.device)
      self.decoder = Decoder(self.latent_dim,
                              self.decoder_hidden,
                              self.output_dim,
                              self.mask,
                              self.scale,
                              self.ref,
                              act_fun=self.act_fun).to(self.device)

    if self.trainable:

      self.optimizer = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()), lr=self.lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.lr_patience, min_lr=1e-6)

      if loss == 'AbsMSE':
        self.loss_fn = nn.MSELoss(reduction='mean')
      elif loss == 'RelMSE':
        self.loss_fn = RelMSELoss

  def forward(self, w):
    '''
    Forward pass through autoencoder.
    input:
    w: (N, self.input_dim) tensor of inputs

    output:
    w_hat: (N, self.input_dim) tensor of outputs that approximate w
    '''
    return self.decoder(self.encoder(w))

  def train(self, batch_size, epochs, n_epochs_print=100, early_stop_patience=200, save_net=True):
    '''
    Train autoencoder and save loss history, best loss, and encoder/decoder weights and biases.

    input:
    batch_size: size of data batches
    epochs:   number of epochs for training
    filename: file name for saving trained encoder and decoder
    n_epochs_print: [optional] number of epochs between printing training and testing loss. Default is 100
    early_stop_patience: [optional] patience for early stopping. Default is 200
    save_net: [optional] Boolean. Set to True to save NN. Default is True
    outputs:
    train_hist_dict: dictionary with fields
              'epoch': number of epochs used in training
              'early_stop_counter': early stop counter at ending epoch
              'valid_loss_hist':  testing loss history
              'train_loss_hist': training loss history
              'best_test_loss': best testing loss
              'best_train_loss': best training loss
              'best_loss_epoch': epoch of best training loss

    autoencoder_dict: dictionary with fields
              'encoder': state dict for best encoder
              'decoder': state dict for best decoder
              'mask':  sparsity mask,
              'scale': scaling vector for data normalization
              'ref':   reference vector for data normalization
              'input_dim': dimension of input layer
              'latent_dim': dimension of latent layer
              'output_dim': dimension of output layer
              'encoder_hidden': dimension of encoder hidden layer
              'decoder_hidden': dimension of decoder hidden layer
              'act_type': activation function used in autoencoder
    '''

    if not self.trainable:
      return {}

    # generate dataloader classes
    train_dataloader = DataLoader(self.train_data, batch_size=batch_size)
    test_dataloader = DataLoader(self.test_data, batch_size=batch_size)

    # train autoencoder
    best_test_loss = np.inf
    best_loss_epoch = 0
    early_stop_counter = 1
    train_loss_hist = []
    valid_loss_hist = []

    start = time()
    for epoch in range(epochs):

      time_epoch_s = time()

      if epoch%n_epochs_print == 0:
        print('\nEpoch {}/{}, Learning rate {}'.format(
          epoch, epochs, self.optimizer.state_dict()['param_groups'][0]['lr']))
        print('-' * 10)
        sys.stdout.flush()

      # train phase
      running_train_loss = 0.0
      self.encoder.train()
      self.decoder.train()
      for batch, (data,) in enumerate(train_dataloader):

        inputs = data.to(self.device)
        outputs = data.to(self.device)

        # prediction and loss
        pred = self.forward(inputs)
        loss = self.loss_fn(outputs, pred)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        running_train_loss += loss.item()*len(inputs)

      epoch_train_loss = running_train_loss/self.train_size
      train_loss_hist.append(epoch_train_loss)
      self.scheduler.step(epoch_train_loss)

      # test phase
      running_test_loss = 0.0
      self.encoder.eval()
      self.decoder.eval()
      for batch, (data,) in enumerate(test_dataloader):
        inputs = data.to(self.device)
        outputs = data.to(self.device)

        with torch.set_grad_enabled(False):
          pred = self.forward(inputs)
          loss = self.loss_fn(outputs, pred)
          running_test_loss += loss.item()*len(inputs)

      epoch_valid_loss = running_test_loss/self.test_size
      valid_loss_hist.append(epoch_valid_loss)

      if epoch_valid_loss < best_test_loss:
        best_test_loss = epoch_valid_loss
        best_train_loss = epoch_train_loss
        best_loss_epoch = epoch
        early_stop_counter = 1
        self.best_encoder = copy.deepcopy(dict(self.encoder.state_dict()))
        self.best_decoder = copy.deepcopy(dict(self.decoder.state_dict()))
      else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
          break

      time_epoch = time() - time_epoch_s

      if epoch % n_epochs_print == 0:
        print('Train loss: {:.5e}'.format(epoch_train_loss))
        print('Valid loss: {:.5e}'.format(epoch_valid_loss))
        print('Exec. time: {:.5e} s'.format(time_epoch))
        sys.stdout.flush()

    train_time = time()-start

    # print out training time and best results
    print()
    if epoch < epochs-1:
      print('Early stopping: {}th training complete in {:.0f}h {:.0f}m {:.0f}s'\
          .format(epoch+1, train_time // 3600, (train_time % 3600) // 60, (train_time % 3600) % 60))
    else:
      print('No early stopping: {}th training complete in {:.0f}h {:.0f}m {:.0f}s'\
          .format(epoch+1, train_time // 3600, (train_time % 3600) // 60, (train_time % 3600) % 60))
    sys.stdout.flush()

    print('\nEpoch {}/{}, Learning rate {}'\
        .format(epoch+1, epochs, self.optimizer.state_dict()['param_groups'][0]['lr']))
    print('-' * 10)
    print('Train loss: {:.5e}'.format(train_loss_hist[-1]))
    print('Valid loss: {:.5e}'.format(valid_loss_hist[-1]))
    sys.stdout.flush()

#         train_hist_dict = { 'epoch': epoch,
#                             'early_stop_counter': early_stop_counter,
#                             'valid_loss_hist': valid_loss_hist,
#                             'train_loss_hist': train_loss_hist,
#                             'best_test_loss': best_test_loss,
#                             'best_train_loss': best_train_loss,
#                             'best_loss_epoch': best_loss_epoch}

    autoencoder_dict = {'encoder': self.best_encoder,
              'decoder': self.best_decoder,
              'mask': self.mask,
              'scale': self.scale,
              'ref': self.ref,
              'input_dim': self.input_dim,
              'latent_dim': self.latent_dim,
              'output_dim': self.output_dim,
              'encoder_hidden': self.encoder_hidden,
              'decoder_hidden': self.decoder_hidden,
              'act_type': self.act_type,
              'train_time': train_time,
              'epoch': epoch,
              'early_stop_counter': early_stop_counter,
              'valid_loss_hist': valid_loss_hist,
              'train_loss_hist': train_loss_hist,
              'best_test_loss': best_test_loss,
              'best_train_loss': best_train_loss,
              'best_loss_epoch': best_loss_epoch}
    if save_net:
      print('Saving net...')
      sys.stdout.flush()
      torch.save(autoencoder_dict, self.filename)
      torch.save(self.state_dict_np(), self.filename[:-2]+"_numpy.p")
      print('Net saved!')
      sys.stdout.flush()

    self.encoder.load_state_dict(self.best_encoder)
    self.decoder.load_state_dict(self.best_decoder)
    return autoencoder_dict


  def state_dict_np(self):
    config_common = {
      "ref": bkd.to_numpy(self.ref).reshape(-1),
      "scale": bkd.to_numpy(self.scale).reshape(-1),
      "input_dim": int(self.input_dim),
      "latent_dim": int(self.latent_dim),
      "activation": str(self.act_type).lower()
    }
    config = {}
    for l in ("encoder", "decoder"):
      hidden = self.decoder_hidden if (l == "decoder") else self.encoder_hidden
      layer = self.best_decoder if (l == "decoder") else self.best_encoder
      mask = self.mask if (l == "decoder") else self.mask.T
      mask_shape = tuple(mask.shape)
      config[l] = {
        "weights": self.get_weights_np(layer, mask_shape),
        "mask_shape": mask_shape,
        "mask_indices": np.vstack(mask.nonzero()),
        "hidden_dim": hidden
      }
      config[l].update(config_common)
    return config

  def get_weights_np(self, weights, mask_shape):
    weights = map_nested_dict(weights, bkd.to_numpy)
    weights_np = {}
    for (i, k) in enumerate((0,2)):
      if (f'net.{k}.indices' in weights):
        mask = tuple([weights[f'net.{k}.indices'][j] for j in range(2)])
        wi = sp.csr_matrix(
          (weights[f'net.{k}.weights'], mask), shape=mask_shape
        )
      else:
        wi = weights[f'net.{k}.weight']
      weights_np[f"W{i+1}"] = wi
    weights_np["b1"] = weights['net.0.bias']
    return weights_np
