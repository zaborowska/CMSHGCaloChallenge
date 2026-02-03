import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import torch
import torch.utils.data as torchdata
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import jetnet
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression


import utils
from plotting.plotting_utils import make_hist

def train_and_evaluate_cls(model, data_train, data_test, optim, arg):
    """ train the model and evaluate along the way"""
    best_eval_acc = float('-inf')
    arg.best_epoch = -1
    try:
        for i in range(arg.cls_n_epochs):
            train_cls(model, data_train, optim, i, arg)
            with torch.no_grad():
                eval_acc, _, _ = evaluate_cls(model, data_test, arg)
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                arg.best_epoch = i+1
                #filename = arg.mode + '_' + arg.dataset + '.pt'
                #torch.save({'model_state_dict':model.state_dict()},
                           #os.path.join(arg.output_dir, filename))
            if eval_acc == 1.:
                break
    except KeyboardInterrupt:
        # training can be cut short with ctrl+c, for example if overfitting between train/test set
        # is clearly visible
        pass
    return model

def train_cls(model, data_train, optim, epoch, arg):
    """ train one step """
    model.train()
    for i, data_batch in enumerate(data_train):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        #input_vector, target_vector = torch.split(data_batch, [data_batch.size()[1]-1, 1], dim=1)
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(output_vector, target_vector.unsqueeze(1))

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % (len(data_train)//2) == 0:
            print('Epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch+1, arg.cls_n_epochs, i, len(data_train), loss.item()))
        # PREDICTIONS
        pred = torch.round(torch.sigmoid(output_vector.detach()))
        target = torch.round(target_vector.detach())
        if i == 0:
            res_true = target
            res_pred = pred
        else:
            res_true = torch.cat((res_true, target), 0)
            res_pred = torch.cat((res_pred, pred), 0)

    try:
        print("Accuracy on training set is",
          accuracy_score(res_true.cpu(), np.clip(res_pred.cpu(), 0., 1.0)))
    except:
        print("Nans")

def evaluate_cls(model, data_test, arg, final_eval=False, calibration_data=None):
    """ evaluate on test set """
    model.eval()
    for j, data_batch in enumerate(data_test):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        pred = output_vector.reshape(-1)
        target = target_vector.double()
        if j == 0:
            result_true = target
            result_pred = pred
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)
    BCE = torch.nn.BCEWithLogitsLoss()(result_pred, result_true)
    result_pred = torch.round(torch.sigmoid(result_pred)).cpu().numpy()
    result_true = result_true.cpu().numpy().astype(np.float32)
    result_pred = np.clip(np.round(result_pred), 0., 1.0)
    #print(np.amin(result_pred), np.amax(result_pred), np.sum(np.isnan(result_pred)))
    try:
        eval_acc = accuracy_score(result_true, result_pred)
    except:
        print("Nans")
        result_pred[np.isnan(result_pred)] = 0.5
        eval_acc = accuracy_score(result_true, result_pred)
    print("Accuracy on test set is", eval_acc)
    eval_auc = roc_auc_score(result_true, result_pred)
    print("AUC on test set is", eval_auc)
    JSD = - BCE + np.log(2.)
    print("BCE loss of test set is {:.4f}, JSD of the two dists is {:.4f}".format(BCE,
                                                                                  JSD/np.log(2.)))
    if final_eval:
        prob_true, prob_pred = calibration_curve(result_true, result_pred, n_bins=10)
        print("unrescaled calibration curve:", prob_true, prob_pred)
        calibrator = calibrate_classifier(model, calibration_data, arg)
        rescaled_pred = calibrator.predict(result_pred)
        eval_acc = accuracy_score(result_true, np.clip(np.round(rescaled_pred), 0., 1.0))
        print("Rescaled accuracy is", eval_acc)
        eval_auc = roc_auc_score(result_true, rescaled_pred)
        print("rescaled AUC of dataset is", eval_auc)
        prob_true, prob_pred = calibration_curve(result_true, rescaled_pred, n_bins=10)
        print("rescaled calibration curve:", prob_true, prob_pred)
        # calibration was done after sigmoid, therefore only BCELoss() needed here:
        BCE = torch.nn.BCELoss()(torch.tensor(rescaled_pred), torch.tensor(result_true))
        JSD = - BCE.cpu().numpy() + np.log(2.)
        otp_str = "rescaled BCE loss of test set is {:.4f}, "+\
            "rescaled JSD of the two dists is {:.4f}"
        print(otp_str.format(BCE, JSD/np.log(2.)))
    return eval_acc, eval_auc, JSD/np.log(2.)

def calibrate_classifier(model, calibration_data, arg):
    """ reads in calibration data and performs a calibration with isotonic regression"""
    model.eval()
    assert calibration_data is not None, ("Need calibration data for calibration!")
    for j, data_batch in enumerate(calibration_data):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        pred = torch.sigmoid(output_vector).reshape(-1)
        target = target_vector.to(torch.float64)
        if j == 0:
            result_true = target
            result_pred = pred
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)
    result_true = result_true.cpu().numpy()
    result_pred = result_pred.cpu().numpy()
    iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1.-1e-6).fit(result_pred,
                                                                                      result_true)
    return iso_reg


class DNN(torch.nn.Module):
    """ NN for vanilla classifier. Does not have sigmoid activation in last layer, should
        be used with torch.nn.BCEWithLogitsLoss()
    """
    def __init__(self, num_layer, num_hidden, input_dim, dropout_probability=0.):
        super(DNN, self).__init__()

        self.dpo = dropout_probability

        self.inputlayer = torch.nn.Linear(input_dim, num_hidden)
        self.outputlayer = torch.nn.Linear(num_hidden, 1)

        all_layers = [self.inputlayer, torch.nn.LeakyReLU(), torch.nn.Dropout(self.dpo)]
        for _ in range(num_layer):
            all_layers.append(torch.nn.Linear(num_hidden, num_hidden))
            all_layers.append(torch.nn.LeakyReLU())
            all_layers.append(torch.nn.Dropout(self.dpo))

        all_layers.append(self.outputlayer)
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        """ Forward pass through the DNN """
        x = self.layers(x)
        return x

def get_feat_names(nLayers):
    feat_names = ['Incident E', 'E Ratio']
    for i in range(nLayers): feat_names.append("Log Energy Layer %i" % i)
    for i in range(nLayers): feat_names.append("X Center Layer %i" % i)
    for i in range(nLayers): feat_names.append("X Width Layer %i" % i)
    for i in range(nLayers): feat_names.append("Y Center Layer %i" % i)
    for i in range(nLayers): feat_names.append("Y Width Layer %i" % i)
    for i in range(nLayers): feat_names.append("Sparsity Layer %i" % i)

    return feat_names


def compute_feats(showers, incident_E, geom):

    eps = 1e-8

    E_total = np.sum(showers, axis=(1,2)).reshape(showers.shape[0], 1)
    E_ratio = E_total / incident_E
    E_per_layer = np.log10( np.sum(showers, axis=(2)) + eps)


    x_vals = geom.xmap[:, :geom.max_ncell]
    E_x_center = utils.WeightedMean(x_vals, showers, axis=(2))
    E_x2_center = utils.WeightedMean(x_vals, showers, power=2, axis=(2))
    E_x_width = utils.GetWidth(E_x_center, E_x2_center)

    y_vals = geom.ymap[:, :geom.max_ncell]
    E_y_center = utils.WeightedMean(y_vals, showers, axis=(2))
    E_y2_center = utils.WeightedMean(y_vals, showers, power=2, axis=(2))
    E_y_width = utils.GetWidth(E_y_center, E_y2_center)

    #r_vals = geom.ring_map[:, :geom.max_ncell]
    #E_R_center = utils.WeightedMean(r_vals, showers, axis=(2))
    #E_R2_center = utils.WeightedMean(r_vals, showers, power=2, axis=(2))
    #E_R_width = utils.GetWidth(E_R_center, E_R2_center)

    #phi_vals = geom.theta_map[:, :geom.max_ncell]
    #E_phi_center, E_phi_width = utils.ang_center_spread(phi_vals, showers, axis=(2))


    eps = 1e-6
    layer_voxels = np.reshape(showers,(showers.shape[0],showers.shape[1],-1))
    layer_sparsity = np.sum(layer_voxels > eps, axis = -1) / layer_voxels.shape[2]

    #feats = np.concatenate([incident_E, E_ratio, E_per_layer, E_x_center, E_x_width, E_y_center, E_y_width], axis = -1).astype(np.float32)
    feats = np.concatenate([incident_E, E_ratio, E_per_layer, E_x_center, E_x_width, E_y_center, E_y_width, layer_sparsity], axis = -1).astype(np.float32)

    return feats


def ttv_split(data1, split=np.array([0.7, 0.2, 0.1])):
    """ splits data1 and data2 in train/test/val according to split,
        returns shuffled and merged arrays
    """
    num_events = (len(data1) * split).astype(int)
    np.random.shuffle(data1)
    train1, test1, val1 = np.split(data1, num_events.cumsum()[:-1])
    np.random.shuffle(train1)
    np.random.shuffle(test1)
    np.random.shuffle(val1)
    return train1, test1, val1


def compute_metrics(flags):

    utils.SetStyle()
    nevts = int(flags.nevts)
    dataset_config = utils.LoadJson(flags.config)
    geom_file = dataset_config.get('BIN_FILE', '')
    dataset_num = dataset_config.get('DATASET_NUM', 2)
    hgcal = dataset_config.get('HGCAL', False)
    max_cells = dataset_config.get('MAX_CELLS', None)

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')
    flags.device = device


    geom = utils.load_geom(geom_file)


    shape_plot = dataset_config['SHAPE_ORIG']

    print("Data shape", shape_plot)

    if(not os.path.exists(flags.plot_folder)): os.system("mkdir -p %s" % flags.plot_folder)


    geom_conv = None



    def LoadFile(fname, EMin = -1.0, nevts = -1):
        print("Load %s" % fname)
        end = None if nevts < 0 else nevts
        scale_fac = 1000.
        with h5.File(fname,"r") as h5f:
            if(hgcal): 
                generated = h5f['showers'][:end,:,:dataset_config['MAX_CELLS']] * scale_fac
                energies = h5f['gen_info'][:end,0] 
            else: 
                generated = h5f['showers'][:end] * scale_fac
                energies = h5f['incident_energies'][:end] * scale_fac

        energies = np.reshape(energies,(-1,1))
        generated = np.reshape(generated,shape_plot)
        if(EMin > 0.):
            mask = generated < EMin
            #print("Applying ECut " + str(EMin))
            #print('before', np.mean(generated))

            #Preserve layer energies after applying threshold
            generated[generated < 0] = 0 
            d_masked = np.where(mask, generated, 0.)
            lostE = np.sum(d_masked, axis = -1, keepdims=True)
            ELayer = np.sum(generated, axis = -1, keepdims=True)
            eps = 1e-10
            rescale = (ELayer + eps)/(ELayer - lostE +eps)
            rescale[ELayer < EMin] = 0.
            generated[mask] = 0.
            generated *= rescale
            #print('after', np.mean(generated))

        return generated,energies

    def LoadSample(fname, EMin = -1.0, nevts = -1, reprocess=False):
        feat_file = fname + ".feat.npz"
        if(os.path.exists(feat_file) and not reprocess):
            print("Load %s" % feat_file)
            feats = np.load(feat_file)['feats']
        else:
            showers, energies = LoadFile(fname, EMin, flags.nevts)
            feats = compute_feats(showers, energies, geom)
            np.savez(feat_file, feats=feats)

        return feats


    geant_energies = None
    geant_showers = None
    feats_gen = feats_geant = None
    data_dict = {}


    if(not flags.geant_only):
        if(flags.generated == ""):
            print("Missing data file to plot!")
            exit(1)
        f_sample_list = utils.get_files(flags.generated)

        for f_sample in f_sample_list: 
            try:
                feats = LoadSample( f_sample, flags.EMin, flags.nevts, reprocess=flags.reprocess)
                if(feats_gen is None): feats_gen = feats
                else: 
                    feats_gen = np.concatenate((feats_gen, feats), axis=0)

                total_evts = feats_gen.shape[0]
                if(flags.nevts > 0 and total_evts >= flags.nevts): break
            except:
                print("Bad file, skipping")

        print("Loaded %i generated showers" % total_evts)


    f_geant_list = utils.get_files(dataset_config['EVAL'], folder=flags.data_folder)
    for f_sample in f_geant_list:
        feats = LoadSample( f_sample, flags.EMin, flags.nevts)

        if(feats_geant is None): feats_geant = feats
        else: feats_geant = np.concatenate((feats_geant, feats), axis=0)

        total_evts = feats_geant.shape[0]
        if(flags.nevts > 0 and total_evts >= flags.nevts): break


    nLayers = shape_plot[1]
    feat_names = get_feat_names(nLayers)

    if(flags.no_sparse):
        #don't include sparsity feature
        feats_no_sparse = [idx for idx,feat_name in enumerate(feat_names) if 'Sparsity' not in feat_name] 
        feats_geant = feats_geant[:, feats_no_sparse]
        feats_gen = feats_gen[:, feats_no_sparse]
        feat_names = [feat_names[idx] for idx in feats_no_sparse]



    do_hists = do_classifier = do_fpd = False

    if(flags.mode == "all"):
        do_hists = do_classifier = do_fpd = True
    elif(flags.mode == "hist"):
        do_hist = True
    elif(flags.mode == "classifier" or flags.mode == "cls"):
        do_classifier = True
    elif(flags.mode == "fpd" or flags.mode == "kpd"):
        do_fpd = True



    #Separation power
    if(do_hists):
        fname = ""
        sep_power_result_str = ""
        sep_power_sum = 0.0
        if feats_gen:
            print(feats_gen.shape)
        else:
            print(feats_geant.shape)
        for i in range(len(feat_names)):
            if(flags.plot): fname = flags.plot_folder + feat_names[i].replace(" ", "") + ".png"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if "Log Energy Layer" in feat_names[i]:
                    # do not show the "0" values in the plot
                    bins = np.linspace(*np.quantile(feats_geant[:, i][feats_geant[:, i] > -8], [0.0, 1.0]), 50)
                else:
                    bins = None
                if(not flags.geant_only):
                    sep_power = make_hist(feats_geant[:,i], feats_gen[:,i], xlabel = feat_names[i], fname =  fname, binning = bins)
                else:
                    sep_power = make_hist(feats_geant[:,i], None, xlabel = feat_names[i], fname =  fname, binning = bins)

            sep_power_sum += sep_power
            sep_power_result_str += "%i %s: %.3e \n" % (i, feat_names[i], sep_power)



        # --- Extra plot: log10(cell energy), with cell energy summed over ALL layers ---
        eps_cell = 1e-8

        # Helper to build flattened log10(sum_layers(cellE)) distribution from a file list
        def _cell_logE_sumLayers(file_list):
            out = None
            total_evts = 0

            for f_sample in file_list:
                # Respect flags.nevts across multiple files
                remaining = -1 if flags.nevts <= 0 else max(flags.nevts - total_evts, 0)
                if remaining == 0:
                    break

                showers, _ = LoadFile(f_sample, flags.EMin, remaining)  # showers: (N, L, C)
                cellE = np.sum(showers, axis=1)                         # (N, C) sum over layers
                cellE = cellE[cellE > 0]
                logE = np.log10(cellE).reshape(-1)

                out = logE if out is None else np.concatenate((out, logE), axis=0)
                total_evts += showers.shape[0]

                if flags.nevts > 0 and total_evts >= flags.nevts:
                    break

            return out

        # Helper to build log10(cellE) distribution from a file list
        def _cell_logE_allCells(file_list):
            out = None
            total_evts = 0

            for f_sample in file_list:
                remaining = -1 if flags.nevts <= 0 else max(flags.nevts - total_evts, 0)
                if remaining == 0:
                    break

                showers, _ = LoadFile(f_sample, flags.EMin, remaining)   # (N, L, C)
                vals = showers[showers > 0]                              # variable length
                logE = np.log10(vals).reshape(-1) if vals.size else np.empty((0,), dtype=float)

                out = logE if out is None else np.concatenate((out, logE), axis=0)
                total_evts += showers.shape[0]

                if flags.nevts > 0 and total_evts >= flags.nevts:
                    break

            return out if out is not None else np.empty((0,), dtype=float)

        # Always build Geant distribution
        cell_logE_geant = _cell_logE_sumLayers(f_geant_list)

        xlabel = "Log10 Cell Energy (sum over layers)"
        fname = ""
        if flags.plot:
            fname = os.path.join(flags.plot_folder, "LogCellEnergySumLayers.png")

        if flags.geant_only:
            make_hist(cell_logE_geant, None, xlabel=xlabel, fname=fname)
        else:
            # Plot Geant vs Gen (two histograms)
            cell_logE_gen = _cell_logE_sumLayers(f_sample_list)

            sep_power_cell = make_hist(
                cell_logE_geant,
                cell_logE_gen,
                xlabel=xlabel,
                fname=fname
            )
            sep_power_sum += sep_power_cell
            sep_power_result_str += "CELL %s: %.3e \n" % (xlabel, sep_power_cell)

        cell_logEcell_geant = _cell_logE_allCells(f_geant_list)
        xlabel = "Log10 Cell Energy (all cells)"
        fname = ""
        cell_logE_bins = np.linspace(*np.quantile(cell_logEcell_geant[cell_logEcell_geant > -8], [0.0, 1.0]), 50)
        if flags.plot:
            fname = os.path.join(flags.plot_folder, "LogAllCellEnergy.png")

        if flags.geant_only:
            make_hist(cell_logEcell_geant, None, xlabel=xlabel, fname=fname)
        else:
            # Plot Geant vs Gen (two histograms)
            cell_logEcell_gen = _cell_logE_allCells(f_sample_list)

            sep_power_cell = make_hist(
                cell_logEcell_geant,
                cell_logEcell_gen,
                xlabel=xlabel,
                fname=fname
            )
            sep_power_sum += sep_power_cell
            sep_power_result_str += "CELL_INDIVIDUAL %s: %.3e \n" % (xlabel, sep_power_cell)
        
            sep_power_result_str += "\n TOTAL : %.2f" % sep_power_sum
            with open(os.path.join(flags.plot_folder, 'sep_power.txt'), 'w') as f:
                f.write(sep_power_result_str)

    #FPD KPD

    if(do_classifier and not flags.geant_only):
        labels_diffu = np.ones((feats_gen.shape[0], 1), dtype=np.float32)
        labels_geant = np.zeros((feats_geant.shape[0], 1), dtype=np.float32)

        labels_all = np.concatenate((labels_diffu, labels_geant), axis = 0)
        feats_all = np.concatenate((feats_gen, feats_geant), axis = 0)

        scaler = StandardScaler()
        feats_all = scaler.fit_transform(feats_all)
        print(feats_all.shape, labels_all.shape)
        inputs_all = np.concatenate((feats_all, labels_all), axis = 1)

        ttv_fracs = np.array([0.7, 0.1, 0.2])
        train_data, test_data, val_data = ttv_split(inputs_all, ttv_fracs)


        input_dim = feats_all.shape[1]
        cls_num_layer = 2
        cls_num_hidden = 2024
        dropout = 0.2
        cls_lr = 1e-4
        classifier = DNN(input_dim = input_dim, num_layer= cls_num_layer, num_hidden = cls_num_hidden, dropout_probability = dropout)
        classifier.to(device)
        print(classifier)
        total_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

        print("Classifier has {} parameters".format( int(total_parameters)))

        optimizer = torch.optim.Adam(classifier.parameters(), lr= cls_lr)


        if flags.save_mem:
            train_data = torchdata.TensorDataset(torch.tensor(train_data))
            test_data = torchdata.TensorDataset(torch.tensor(test_data))
            val_data = torchdata.TensorDataset(torch.tensor(val_data))
        else:
            train_data = torchdata.TensorDataset(torch.tensor(train_data).to(device))
            test_data = torchdata.TensorDataset(torch.tensor(test_data).to(device))
            val_data = torchdata.TensorDataset(torch.tensor(val_data).to(device))

        train_dataloader = torchdata.DataLoader(train_data, batch_size=flags.cls_batch_size, shuffle=True)
        test_dataloader = torchdata.DataLoader(test_data, batch_size=flags.cls_batch_size, shuffle=False)
        val_dataloader = torchdata.DataLoader(val_data, batch_size=flags.cls_batch_size, shuffle=False)

        for i in range(flags.cls_n_iters):
            classifier = train_and_evaluate_cls(classifier, train_dataloader, val_dataloader, optimizer, flags)
            #classifier = load_classifier(classifier, flags)

            with torch.no_grad():
                print("Now looking at independent dataset:")
                eval_acc, eval_auc, eval_JSD = evaluate_cls(classifier, test_dataloader, flags,
                                                            final_eval=True,
                                                            calibration_data=val_dataloader)
            print("Final result of classifier test (AUC / JSD):")
            print("{:.4f} / {:.4f}".format(eval_auc, eval_JSD))
            with open(os.path.join(flags.plot_folder, 'metrics.txt'), 'a') as f:
                f.write('Final result of classifier test (AUC / JSD):\n'+\
                        '{:.4f} / {:.4f}\n\n'.format(eval_auc, eval_JSD))

    if(do_fpd and not flags.geant_only):
        min_samples = min(feats_geant.shape[0], 20000)
        fpd_val, fpd_err = jetnet.evaluation.fpd(feats_geant, feats_gen, min_samples = min_samples)
        kpd_val, kpd_err = jetnet.evaluation.kpd(feats_geant, feats_gen)

        fpd_result_str = (
                f"FPD (x10^3): {fpd_val*1e3:.4f} ± {fpd_err*1e3:.4f}\n" 
                f"KPD (x10^3): {kpd_val*1e3:.4f} ± {kpd_err*1e3:.4f}\n"
            )
        print(fpd_result_str)
        with open(os.path.join(flags.plot_folder, 'metrics.txt'), 'a') as f:
            f.write(fpd_result_str)


if(__name__ == "__main__"):

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--plot_folder', default='plots/eval/', help='Folder to save results')
    parser.add_argument('-d', '--data_folder', default='data_dir/', help='Folder with Geant dataset')
    parser.add_argument('--generated', '-g', default='', help='Generated showers')
    parser.add_argument('--config', '-c', default='config_dataset2.json', help='Training parameters')
    parser.add_argument('-n', '--nevts', type=int,default=-1, help='Number of events to load')
    parser.add_argument('--EMin', type = float, default=-1.0, help='Voxel min energy')

    parser.add_argument('--plot', default=False, action='store_true', help='Save 1D feature plots')

    parser.add_argument('--cls_n_iters', default=1, type=int, help='Num classifiers to train')
    parser.add_argument('--cls_n_epochs', default=50, type=int, help='Num classifier epochs')
    parser.add_argument('--cls_batch_size', default=256, type=int, help='classifier batch size')
    parser.add_argument('--save_mem', action='store_true', default=False,help='Limit GPU memory')

    parser.add_argument('--geant_only', action='store_true', default=False,help='Plots with just geant')
    parser.add_argument('--reprocess', action='store_true', default=False,help='Recompute features for eval')
    parser.add_argument('--no_sparse', action='store_true', default=False,help='Dont include sparsity feature')
    parser.add_argument('-m', '--mode', default='all', help='Which eval metrics to run. Options : hist, cls, fpd, all (default)')

    flags = parser.parse_args()
    compute_metrics(flags)
