import torch
from torch.nn import functional as F
from linear_nets import MLP,fc_layer, Drebin_MLP_Net, Drebin_Classifier
from exemplars import ExemplarHandler
from continual_learner import ContinualLearner
from replayer import Replayer
import utils



from torch import nn
import numpy as np
import utils
import excitability_modules as em


class Ember_MLP_Net_V2(nn.Module):
    def __init__(self):
        super(Ember_MLP_Net_V2, self).__init__()
        
        input_features = 2381
        self.fc1 = nn.Linear(input_features, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.act1 = nn.ReLU()
        #self.fc1_drop = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.act2 = nn.ReLU()
        #self.fc2_drop = nn.Dropout(p=0.5)
        
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.act3 = nn.ReLU()
        #self.fc3_drop = nn.Dropout(p=0.5)        
        
        self.fc4 = nn.Linear(256, 128)
        self.fc4_bn = nn.BatchNorm1d(128)
        self.act4 = nn.ReLU()
        #self.fc4_drop = nn.Dropout(p=0.5)  
        
        #self.fc_last = nn.Linear(128, 1) 
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc1(x)
        #x = self.fc1_bn(x)
        x = self.act1(x) 
        #x = self.fc1_drop(x)

        x = self.fc2(x)
        #x = self.fc2_bn(x)
        x = self.act2(x) 
        #x = self.fc2_drop(x)
        
        x = self.fc3(x)
        #x = self.fc3_bn(x)
        x = self.act3(x) 
        #x = self.fc3_drop(x)
        
        x = self.fc4(x)
        #x = self.fc4_bn(x)
        x = self.act4(x)
        #x = self.fc4_drop(x)
        
        #x = self.fc_last(x)
        #x = self.out(x)
        return x


class EMBER_Classifier(nn.Module):
    def __init__(self, input_feats_length, output_classes):
        super(EMBER_Classifier, self).__init__()
        self.input_feats_length = input_feats_length
        self.output_classes = output_classes
        
        self.fc_last = nn.Linear(self.input_feats_length, self.output_classes) 

    def forward(self, x):
        x = self.fc_last(x)
        return x  
    
    
class fc_layer(nn.Module):
    '''Fully connected layer, with possibility of returning "pre-activations".

    Input:  [batch_size] x ... x [in_size] tensor
    Output: [batch_size] x ... x [out_size] tensor'''

    def __init__(self, in_size, out_size, nl=nn.ReLU(),
                 drop=0., bias=True, excitability=False, excit_buffer=False, batch_norm=False, gated=False):
        super().__init__()
        if drop>0:
            self.dropout = nn.Dropout(drop)
        self.linear = em.LinearExcitability(in_size, out_size, bias=False if batch_norm else bias,
                                            excitability=excitability, excit_buffer=excit_buffer)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_size)
        if gated:
            self.gate = nn.Linear(in_size, out_size)
            self.sigmoid = nn.Sigmoid()
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl=="none":
            self.nl = nn.ReLU() if nl == "relu" else (nn.LeakyReLU() if nl == "leakyrelu" else utils.Identity())

    def forward(self, x, return_pa=False):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        #pre_activ = self.bn(self.linear(input)) if hasattr(self, 'bn') else self.linear(input)
        
        pre_activ = self.linear(input)

        gate = self.sigmoid(self.gate(x)) if hasattr(self, 'gate') else None
        gated_pre_activ = gate * pre_activ if hasattr(self, 'gate') else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, 'nl') else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        return [self.linear, self.gate] if hasattr(self, 'gate') else [self.linear]


class fc_layer_split(nn.Module):
    '''Fully connected layer outputting [mean] and [logvar] for each unit.

    Input:  [batch_size] x ... x [in_size] tensor
    Output: tuple with two [batch_size] x ... x [out_size] tensors'''

    def __init__(self, in_size, out_size, nl_mean=nn.Sigmoid(), nl_logvar=nn.Hardtanh(min_val=-4.5, max_val=0.),
                 drop=0., bias=True, excitability=False, excit_buffer=False, batch_norm=False, gated=False):
        super().__init__()

        self.mean = fc_layer(in_size, out_size, drop=drop, bias=bias, excitability=excitability,
                             excit_buffer=excit_buffer, batch_norm=batch_norm, gated=gated, nl=nl_mean)
        self.logvar = fc_layer(in_size, out_size, drop=drop, bias=False, excitability=excitability,
                               excit_buffer=excit_buffer, batch_norm=batch_norm, gated=gated, nl=nl_logvar)

    def forward(self, x):
        return (self.mean(x), self.logvar(x))

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.mean.list_init_layers()
        list += self.logvar.list_init_layers()
        return list

#-----------------------------------------------------------------------------------------------------------#


class MLPwoBN(nn.Module):
    '''Module for a multi-layer perceptron (MLP).

    Input:  [batch_size] x ... x [size_per_layer[0]] tensor
    Output: (tuple of) [batch_size] x ... x [size_per_layer[-1]] tensor'''

    def __init__(self, input_size=1000, output_size=10, layers=2, hid_size=1000, hid_smooth=None, size_per_layer=None,
                 drop=0, batch_norm=False, nl="relu", bias=True, excitability=False, excit_buffer=False, gated=False,
                 output='normal'):
        '''sizes: 0th=[input], 1st=[hid_size], ..., 1st-to-last=[hid_smooth], last=[output].
        [input_size]       # of inputs
        [output_size]      # of units in final layer
        [layers]           # of layers
        [hid_size]         # of units in each hidden layer
        [hid_smooth]       if None, all hidden layers have [hid_size] units, else # of units linearly in-/decreases s.t.
                             final hidden layer has [hid_smooth] units (if only 1 hidden layer, it has [hid_size] units)
        [size_per_layer]   None or <list> with for each layer number of units (1st element = number of inputs)
                                --> overwrites [input_size], [output_size], [layers], [hid_size] and [hid_smooth]
        [drop]             % of each layer's inputs that is randomly set to zero during training
        [batch_norm]       <bool>; if True, batch-normalization is applied to each layer
        [nl]               <str>; type of non-linearity to be used (options: "relu", "leakyrelu", "none")
        [gated]            <bool>; if True, each linear layer has an additional learnable gate
        [output]           <str>; if - "normal", final layer is same as all others
                                     - "BCE", final layer has sigmoid non-linearity'''

        super().__init__()
        self.output = output

        # get sizes of all layers
        if size_per_layer is None:
            hidden_sizes = []
            if layers > 1:
                if (hid_smooth is not None):
                    hidden_sizes = [int(x) for x in np.linspace(hid_size, hid_smooth, num=layers-1)]
                else:
                    hidden_sizes = [int(x) for x in np.repeat(hid_size, layers - 1)]
            size_per_layer = [input_size] + hidden_sizes + [output_size]
        self.layers = len(size_per_layer)-1

        # set label for this module
        # -determine "non-default options"-label
        nd_label = "{drop}{bias}{exc}{bn}{nl}{gate}{out}".format(
            drop="" if drop==0 else "-drop{}".format(drop),
            bias="" if bias else "-noBias", exc="-exc" if excitability else "", bn="-bn" if batch_norm else "",
            nl="-lr" if nl=="leakyrelu" else "", gate="-gated" if gated else "",
            out="" if output=="normal" else "-{}".format(output),
        )
        # -set label
        self.label = "MLP({}{})".format(size_per_layer, nd_label) if self.layers>0 else ""

        # set layers
        for lay_id in range(1, self.layers+1):
            # number of units of this layer's input and output
            in_size = size_per_layer[lay_id-1]
            out_size = size_per_layer[lay_id]
            # define and set the fully connected layer
            if lay_id==self.layers and output in ("logistic", "gaussian"):
                layer = fc_layer_split(
                    in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer, drop=drop,
                    batch_norm=False, gated=gated,
                    nl_mean=nn.Sigmoid() if output=="logistic" else utils.Identity(),
                    nl_logvar=nn.Hardtanh(min_val=-4.5, max_val=0.) if output=="logistic" else utils.Identity(),
                )
            else:
                layer = fc_layer(
                    in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer, drop=drop,
                    batch_norm=False if (lay_id==self.layers and not output=="normal") else batch_norm, gated=gated,
                    nl=nn.Sigmoid() if (lay_id==self.layers and not output=="normal") else nl,
                )
            setattr(self, 'fcLayer{}'.format(lay_id), layer)

        # if no layers, add "identity"-module to indicate in this module's representation nothing happens
        if self.layers<1:
            self.noLayers = utils.Identity()

    def forward(self, x):
        for lay_id in range(1, self.layers+1):
            x = getattr(self, 'fcLayer{}'.format(lay_id))(x)
        return x

    @property
    def name(self):
        return self.label

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        for layer_id in range(1, self.layers+1):
            list += getattr(self, 'fcLayer{}'.format(layer_id)).list_init_layers()
        return list


    
    

class Classifier(ContinualLearner, Replayer, ExemplarHandler):
    '''Model for classifying images, "enriched" as "ContinualLearner"-, Replayer- and ExemplarHandler-object.'''

    def __init__(self, image_size, image_channels, classes,
                 fc_layers=4, fc_units=1000, fc_drop=0., fc_bn=False, fc_nl="relu", gated=False,
                 bias=True, excitability=False, excit_buffer=False, binaryCE=False, binaryCE_distill=False, AGEM=False):

        # configurations
        super().__init__()
        self.classes = classes
        self.label = "Classifier"
        self.fc_layers = fc_layers
        self.fc_drop = fc_drop

        # settings for training
        self.binaryCE = binaryCE                 #-> use binary (instead of multiclass) prediction error
        self.binaryCE_distill = binaryCE_distill #-> for classes from previous tasks, use the by the previous model
                                                 #   predicted probs as binary targets (only in Class-IL with binaryCE)

        # check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")


        ######------SPECIFY MODEL------######

        # flatten image to 2D-tensor
        self.flatten = utils.Flatten()
        
        sizes_layers = [2381, 1024, 512, 256, 128]
        
        #[1024, 512, 256, 128]
        # fully connected hidden layers
        
        
        self.fcE = Ember_MLP_Net_V2()
        
        mlp_output_size = fc_units if fc_layers>1 else image_channels*image_size**2

        # classifier
        self.classifier = EMBER_Classifier(128, classes)


    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list

    @property
    def name(self):
        return "{}_c{}".format(self.fcE.name, self.classes)


    def forward(self, x):
        final_features = self.fcE(self.flatten(x))
        return self.classifier(final_features)

    def feature_extractor(self, images):
        return self.fcE(self.flatten(images))


    def train_a_batch(self, x, y, scores=None, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None, task=1):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <tensor> batch of corresponding labels
        [scores]          None or <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x]
                            NOTE: only to be used for "BCE with distill" (only when scenario=="class")
        [x_]              None or (<list> of) <tensor> batch of replayed inputs
        [y_]              None or (<list> of) <tensor> batch of corresponding "replayed" labels
        [scores_]         None or (<list> of) <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]             <number> in [0,1], relative importance of new task
        [active_classes]  None or (<list> of) <list> with "active" classes
        [task]            <int>, for setting task-specific mask'''

        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        # Should gradient be computed separately for each task? (needed when a task-mask is combined with replay)
        gradient_per_task = True if ((self.mask_dict is not None) and (x_ is not None)) else False


        ##--(1)-- REPLAYED DATA --##

        if x_ is not None:
            # In the Task-IL scenario, [y_] or [scores_] is a list and [x_] needs to be evaluated on each of them
            # (in case of 'exact' or 'exemplar' replay, [x_] is also a list!
            TaskIL = (type(y_)==list) if (y_ is not None) else (type(scores_)==list)
            if not TaskIL:
                y_ = [y_]
                scores_ = [scores_]
                active_classes = [active_classes] if (active_classes is not None) else None
            n_replays = len(y_) if (y_ is not None) else len(scores_)

            # Prepare lists to store losses for each replay
            loss_replay = [None]*n_replays
            predL_r = [None]*n_replays
            distilL_r = [None]*n_replays

            # Run model (if [x_] is not a list with separate replay per task and there is no task-specific mask)
            if (not type(x_)==list) and (self.mask_dict is None):
                y_hat_all = self(x_)

            # Loop to evalute predictions on replay according to each previous task
            for replay_id in range(n_replays):

                # -if [x_] is a list with separate replay per task, evaluate model on this task's replay
                if (type(x_)==list) or (self.mask_dict is not None):
                    x_temp_ = x_[replay_id] if type(x_)==list else x_
                    if self.mask_dict is not None:
                        self.apply_XdGmask(task=replay_id+1)
                    y_hat_all = self(x_temp_)

                # -if needed (e.g., Task-IL or Class-IL scenario), remove predictions for classes not in replayed task
                y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]

                # Calculate losses
                if (y_ is not None) and (y_[replay_id] is not None):
                    if self.binaryCE:
                        binary_targets_ = utils.to_one_hot(y_[replay_id].cpu(), y_hat.size(1)).to(y_[replay_id].device)
                        predL_r[replay_id] = F.binary_cross_entropy_with_logits(
                            input=y_hat, target=binary_targets_, reduction='none'
                        ).sum(dim=1).mean()     #--> sum over classes, then average over batch
                    else:
                        predL_r[replay_id] = F.cross_entropy(y_hat, y_[replay_id], reduction='mean')
                if (scores_ is not None) and (scores_[replay_id] is not None):
                    # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes are added to [scores]!
                    n_classes_to_consider = y_hat.size(1)    #--> zeros will be added to [scores] to make it this size!
                    kd_fn = utils.loss_fn_kd_binary if self.binaryCE else utils.loss_fn_kd
                    distilL_r[replay_id] = kd_fn(scores=y_hat[:, :n_classes_to_consider],
                                                 target_scores=scores_[replay_id], T=self.KD_temp)
                # Weigh losses
                if self.replay_targets=="hard":
                    loss_replay[replay_id] = predL_r[replay_id]
                elif self.replay_targets=="soft":
                    loss_replay[replay_id] = distilL_r[replay_id]

                # If needed, perform backward pass before next task-mask (gradients of all tasks will be accumulated)
                if gradient_per_task:
                    weight = 1 if self.AGEM else (1 - rnt)
                    weighted_replay_loss_this_task = weight * loss_replay[replay_id] / n_replays
                    weighted_replay_loss_this_task.backward()

        # Calculate total replay loss
        loss_replay = None if (x_ is None) else sum(loss_replay) / n_replays




        ##--(2)-- CURRENT DATA --##

        if x is not None:
            # If requested, apply correct task-specific mask
            if self.mask_dict is not None:
                self.apply_XdGmask(task=task)

            # Run model
            y_hat = self(x)
            #print(y, y_hat)
            #print(f'active classes in encoder.py {active_classes}')
            
            # -if needed, remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                y_hat = y_hat[:, class_entries]

            # Calculate prediction loss
            if self.binaryCE:
                # -binary prediction loss
                binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
                if self.binaryCE_distill and (scores is not None):
                    classes_per_task = int(y_hat.size(1) / task)
                    binary_targets = binary_targets[:, -(classes_per_task):]
                    binary_targets = torch.cat([torch.sigmoid(scores / self.KD_temp), binary_targets], dim=1)
                predL = None if y is None else F.binary_cross_entropy_with_logits(
                    input=y_hat, target=binary_targets, reduction='none'
                ).sum(dim=1).mean()     #--> sum over classes, then average over batch
            else:
                # -multiclass prediction loss
                predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='mean')

            # Weigh losses
            loss_cur = predL
            #import numpy as np
            #print(y)
            # Calculate training-precision
            #print(y_hat)
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)

            # If backward passes are performed per task (e.g., XdG combined with replay), perform backward pass
            if gradient_per_task:
                weighted_current_loss = rnt*loss_cur
                weighted_current_loss.backward()
        else:
            precision = predL = None
            # -> it's possible there is only "replay" [e.g., for offline with task-incremental learning]


        # Combine loss from current and replayed batch
        if x_ is None:
            loss_total = loss_cur
        else:
            loss_total = loss_replay if (x is None) else rnt*loss_cur+(1-rnt)*loss_replay




        # Backpropagate errors (if not yet done)
        if not gradient_per_task:
            loss_total.backward()



        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
            'pred': predL.item() if predL is not None else 0,
            'pred_r': sum(predL_r).item()/n_replays if (x_ is not None and predL_r[0] is not None) else 0,
            'distil_r': sum(distilL_r).item()/n_replays if (x_ is not None and distilL_r[0] is not None) else 0,
            'precision': precision if precision is not None else 0.,
        }

