#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os,re,sys,math
import argparse
from multiprocessing import Pool
from functools import partial
import numpy as np
import scipy.sparse
import torch
import pandas as pd
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map

def fingerprint(smiles_or_mol, fp_type='morgan', dtype=None, morgan__r=2,
                morgan__n=1024, *args, **kwargs):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits

    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n),
                                 dtype='uint8')
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint

def fingerprints(smiles_mols_array, n_jobs=1, already_unique=False, *args,
                 **kwargs):
        '''
        Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
        e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
        Inserts np.NaN to rows corresponding to incorrect smiles.
        IMPORTANT: if there is at least one np.NaN, the dtype would be float
        Parameters:
            smiles_mols_array: list/array/pd.Series of smiles or already computed
                RDKit molecules
            n_jobs: number of parralel workers to execute
            already_unique: flag for performance reasons, if smiles array is big
                and already unique. Its value is set to True if smiles_mols_array
                contain RDKit molecules already.
        '''
        if isinstance(smiles_mols_array, pd.Series):
            smiles_mols_array = smiles_mols_array.values
        else:
            smiles_mols_array = np.asarray(smiles_mols_array)
        if not isinstance(smiles_mols_array[0], str):
            already_unique = True

        if not already_unique:
            smiles_mols_array, inv_index = np.unique(smiles_mols_array,
                                                     return_inverse=True)

        fps = mapper(n_jobs)(
            partial(fingerprint, *args, **kwargs), smiles_mols_array
        )

        length = 1
        for fp in fps:
            if fp is not None:
                length = fp.shape[-1]
                first_fp = fp
                break
        fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
               for fp in fps]
        if scipy.sparse.issparse(first_fp):
            fps = scipy.sparse.vstack(fps).tocsr()
        else:
            fps = np.vstack(fps)
        if not already_unique:
            return fps[inv_index]
        return fps

def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1, out_smi=False):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    record_stock = set()
    record_gen = set()
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if out_smi:
                stock_max = jac.max(1)
                gen_max = jac.max(0)
                for k in range(0,len(stock_max)):
                    if stock_max[k] > 0.8:
                        record_stock.add(k+j)
                for n in range(0,len(gen_max)):
                    if gen_max[n] > 0.8:
                        record_gen.add(n+i)
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    if out_smi:
        return np.mean(agg_tanimoto), record_stock, record_gen
    else:
        return np.mean(agg_tanimoto)


class Metric:
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None, out_smi=False):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen, out_smi)

    def precalc(self, moleclues):
        raise NotImplementedError

    def metric(self, pref, pgen, out_smi):
        raise NotImplementedError

def internal_diversity(gen, n_jobs=1, device='cpu', fp_type='morgan',
                       gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps,
                                     agg='mean', device=device, p=p)).mean()


class SNNMetric(Metric):
    """
    Computes average max similarities of gen SMILES to ref SMILES
    """
    def __init__(self, fp_type='morgan', out_smi=False, **kwargs):
        self.fp_type = fp_type
        self.out_smi = out_smi
        super().__init__(**kwargs)

    def precalc(self, mols):
        return {'fps': fingerprints(mols, n_jobs=self.n_jobs,
                                    fp_type=self.fp_type)}

    def metric(self, pref, pgen, out_smi):
        return average_agg_tanimoto(pref['fps'], pgen['fps'], out_smi=out_smi,
                                    device=self.device)

def read_smiles_csv(path):
    return pd.read_csv(path,
                       usecols=['SMILES'],skip_blank_lines=True,
                       squeeze=True).astype(str).tolist()

def read_transfer_csv(path):
    return pd.read_csv(path,
                       usecols=['SMILES', 'LEVEL'],
                       ).astype(str)

def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def Rediscovery(gen, goal, n_jobs=1):
    gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    gen_smiles_set = set(gen_smiles) - {None}
    goal_set = set(goal)
    same_set = gen_smiles_set & goal_set
    return len(same_set) / len(gen_smiles_set), same_set

def max_similarity(holdout_smi, dataset_smi):
    #return the smiles with the max similarity from dataset_smi for holdout_smi
    dataset_fps = []
    for smiles_dataset in dataset_smi:
        try:
            m = Chem.MolFromSmiles(smiles_dataset)
            fps = AllChem.GetMorganFingerprintAsBitVect(m, 2, 4096)
            dataset_fps.append(fps)
        except:
            pass
    max_sim_list = []
    max_sim_smiles_list = []
    for smiles_holdout in holdout_smi:
        try:
            m = Chem.MolFromSmiles(smiles_holdout)
            fps_dataset = AllChem.GetMorganFingerprintAsBitVect(m, 2, 4096)
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fps_dataset, dataset_fps))
            max_tanimoto = sims.max()
            max_index = sims.argmax()
            max_sim_list.append(max_tanimoto)
            max_sim_smiles_list.append(dataset_smi[max_index])
        except:
            pass
    return max_sim_list, max_sim_smiles_list

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path',type=str,
                        help='Path to fine-tuning molecules csv')
    parser.add_argument('--goal_path',type=str,
                        help='Path to target molecules csv')
    parser.add_argument('--gen_path',type=str,
                        help='Path to generated molecules csv')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of threads')
    parser.add_argument('--metrics_path', type=str, default='metrics.csv',
                        help='Path to output file with metrics')

    return parser

def main(config):
    csv_train = config.train_path
    csv_goal = config.goal_path
    smi_gen = config.gen_path
    n_jobs = config.n_jobs
    metrics_out = config.metrics_path
    kwargs = {'n_jobs': n_jobs}

    #read files
    df_train = read_transfer_csv(csv_train)
    smi_train = read_smiles_csv(csv_train)
    train = mapper(n_jobs)(canonic_smiles, smi_train)
    train_mols = mapper(n_jobs)(get_mol, train)
    ptrain = {}
    ptrain['SNN'] = SNNMetric(**kwargs).precalc(train_mols)

    df_goal = read_transfer_csv(csv_goal)
    smi_goal = read_smiles_csv(csv_goal)
    goal = mapper(n_jobs)(canonic_smiles, smi_goal)
    goal_mols = mapper(n_jobs)(get_mol, goal)
    pgoal = {}
    pgoal['SNN'] = SNNMetric(**kwargs).precalc(goal_mols)

    smi_gen = read_smiles_csv(smi_gen)
    gen = mapper(n_jobs)(canonic_smiles, smi_gen)
    gen_mols = mapper(n_jobs)(get_mol, gen)
    pgen = {}
    pgen['SNN'] = SNNMetric(**kwargs).precalc(gen_mols)

    # Metrics
    metrics = {}
    metrics['IntDiv'] = internal_diversity(gen_mols, n_jobs)
    metrics['SNN/Gen_train'] = SNNMetric(**kwargs)(
        pgen=pgen['SNN'], pref=ptrain['SNN'])
    # metrics['IntDiv_goal'] = internal_diversity(goal_mols, n_jobs)
    metrics['SNN/Gen_goal'] = SNNMetric(**kwargs)(
        pgen=pgen['SNN'], pref=pgoal['SNN'])
    Rediscovery_value, same_smi_set = Rediscovery(gen, goal, n_jobs)
    same_smi_list = list(same_smi_set)
    same_max_sim_list = []
    same_max_sim_smiles_list = []
    re_goal_train_07_number = 0
    gen_goal_07_num = 0
    gen_goal_train_07_number = 0
    gen_goal_08_num = 0
    gen_goal_train_08_number = 0
    gen_goal_09_num = 0
    gen_goal_train_09_number = 0
    level_list = ['A', 'B', 'C', 'D', 'E']
    same_smi_level_list = []
    same_smi_level_07_list = []
    same_max_sim_smiles_level_list = []
    metrics['IntDiv_Rediscovery'] = '/'  # if the model cant reproduce a molecule, the value is /
    metrics['SNN/Rediscovery_train'] = '/'  # if the model cant reproduce a molecule, the value is /
    metrics['Rediscovery'] = 0
    metrics['Rediscovery_number'] = 0
    for level_item in level_list:
        metrics['Rediscovery_%s' % level_item] = 0
    metrics['Rediscovery_0.7'] = 0
    metrics['Rediscovery_0.7_number'] = 0
    for level_item in level_list:
        metrics['Rediscovery_0.7_%s_number' % level_item] = 0
    if len(same_smi_list) > 0:
        for smiles in same_smi_list:
            level = df_goal[df_goal['SMILES'] == smiles]['LEVEL'].values
            same_smi_level_list.extend(level)
        psamegoal = {}
        same_smi_mols = mapper(n_jobs)(get_mol, same_smi_list)
        psamegoal['SNN'] = SNNMetric(**kwargs).precalc(same_smi_mols)
        metrics['IntDiv_Rediscovery'] = internal_diversity(same_smi_mols, n_jobs)
        metrics['SNN/Rediscovery_train'] = SNNMetric(**kwargs)(
            pgen=psamegoal['SNN'], pref=ptrain['SNN'])
        same_max_sim_list, same_max_sim_smiles_list = max_similarity(same_smi_list, train)
        for smiles in same_max_sim_smiles_list:
            level = df_train[df_train['SMILES'] == smiles]['LEVEL'].values
            same_max_sim_smiles_level_list.extend(level)
        for i in range(len(same_max_sim_list)):
            if same_max_sim_list[i] < 0.7:
                level = df_goal[df_goal['SMILES'] == same_smi_list[i]]['LEVEL'].values
                same_smi_level_07_list.extend(level)
                re_goal_train_07_number += 1
        metrics['Rediscovery'] = Rediscovery_value
        metrics['Rediscovery_number'] = len(same_smi_list)
        for level_item in level_list:
            metrics['Rediscovery_%s' % level_item] = same_smi_level_list.count(level_item)
        metrics['Rediscovery_0.7'] = \
            re_goal_train_07_number / len(same_smi_list)
        metrics['Rediscovery_0.7_number'] = re_goal_train_07_number
        for level_item in level_list:
            metrics['Rediscovery_0.7_%s_number' % level_item] = same_smi_level_07_list.count(level_item)
    gen_train_max_sim_list, gen_train_max_sim_smiles_list = max_similarity(gen, train)
    gen_goal_max_sim_list, gen_goal_max_sim_smiles_list = max_similarity(gen, goal)
    gen_goal_train_max_sim_list, gen_goal_train_max_sim_smiles_list = max_similarity(gen_goal_max_sim_smiles_list,
                                                                                     train)
    gen_train_max_sim_smiles_level_list = []
    gen_goal_max_sim_smiles_level_list = []
    gen_goal_train_max_sim_smiles_level_list = []
    for smiles in gen_train_max_sim_smiles_list:
        level = df_train[df_train['SMILES'] == smiles]['LEVEL'].values
        gen_train_max_sim_smiles_level_list.extend(level)
    for smiles in gen_goal_max_sim_smiles_list:
        level = df_goal[df_goal['SMILES'] == smiles]['LEVEL'].values
        gen_goal_max_sim_smiles_level_list.extend(level)
    for smiles in gen_goal_train_max_sim_smiles_list:
        level = df_train[df_train['SMILES'] == smiles]['LEVEL'].values
        gen_goal_train_max_sim_smiles_level_list.extend(level)
    gen_goal_07_level_list = []
    gen_goal_07_train_07_level_list = []
    gen_goal_08_level_list = []
    gen_goal_08_train_07_level_list = []
    gen_goal_09_level_list = []
    gen_goal_09_train_07_level_list = []
    for i in range(len(gen_goal_max_sim_list)):
        if gen_goal_max_sim_list[i] > 0.7:
            gen_goal_07_num += 1
            level = df_goal[df_goal['SMILES'] == gen_goal_max_sim_smiles_list[i]][
                'LEVEL'].values
            gen_goal_07_level_list.extend(level)
            if gen_train_max_sim_list[i] < 0.7:
                gen_goal_train_07_number += 1
                level = df_goal[
                    df_goal['SMILES'] == gen_goal_max_sim_smiles_list[i]][
                    'LEVEL'].values
                gen_goal_07_train_07_level_list.extend(level)
        if gen_goal_max_sim_list[i] > 0.8:
            gen_goal_08_num += 1
            level = df_goal[df_goal['SMILES'] == gen_goal_max_sim_smiles_list[i]][
                'LEVEL'].values
            gen_goal_08_level_list.extend(level)
            if gen_train_max_sim_list[i] < 0.7:
                gen_goal_train_08_number += 1
                level = df_goal[
                    df_goal['SMILES'] == gen_goal_max_sim_smiles_list[i]][
                    'LEVEL'].values
                gen_goal_08_train_07_level_list.extend(level)
        if gen_goal_max_sim_list[i] > 0.9:
            gen_goal_09_num += 1
            level = df_goal[df_goal['SMILES'] == gen_goal_max_sim_smiles_list[i]][
                'LEVEL'].values
            gen_goal_09_level_list.extend(level)
            if gen_train_max_sim_list[i] < 0.7:
                gen_goal_train_09_number += 1
                level = df_goal[
                    df_goal['SMILES'] == gen_goal_max_sim_smiles_list[i]][
                    'LEVEL'].values
                gen_goal_09_train_07_level_list.extend(level)

    metrics['Sim_0.7'] = 0
    metrics['Sim_0.7_number'] = 0
    for level_item in level_list:
        metrics['Sim_0.7_%s' % level_item] = 0
    metrics['Sim_0.7_train_0.7'] = 0
    metrics['Sim_0.7_train_0.7_number'] = 0
    for level_item in level_list:
        metrics['Sim_0.7_train_0.7_%s_number' % level_item] = 0
    metrics['Sim_0.8'] = 0
    metrics['Sim_0.8_number'] = 0
    for level_item in level_list:
        metrics['Sim_0.8_%s' % level_item] = 0
    metrics['Sim_0.8_train_0.7'] = 0
    metrics['Sim_0.8_train_0.7_number'] = 0
    for level_item in level_list:
        metrics['Sim_0.8_train_0.7_%s_number' % level_item] = 0
    metrics['Sim_0.9'] = 0
    metrics['Sim_0.9_number'] = 0
    for level_item in level_list:
        metrics['Sim_0.9_%s' % level_item] = 0
    metrics['Sim_0.9_train_0.7'] = 0
    metrics['Sim_0.9_train_0.7_number'] = 0
    for level_item in level_list:
        metrics['Sim_0.9_train_0.7_%s_number' % level_item] = 0

    if gen_goal_07_num > 0:
        metrics['Sim_0.7'] = gen_goal_07_num / len(gen)
        metrics['Sim_0.7_number'] = gen_goal_07_num
        for level_item in level_list:
            metrics['Sim_0.7_%s' % level_item] = gen_goal_07_level_list.count(level_item)
        metrics['Sim_0.7_train_0.7'] = gen_goal_train_07_number / gen_goal_07_num
        metrics['Sim_0.7_train_0.7_number'] = gen_goal_train_07_number
        for level_item in level_list:
            metrics['Sim_0.7_train_0.7_%s_number' % level_item] = gen_goal_07_train_07_level_list.count(level_item)
    if gen_goal_08_num > 0:
        metrics['Sim_0.8'] = gen_goal_08_num / len(gen)
        metrics['Sim_0.8_number'] = gen_goal_08_num
        for level_item in level_list:
            metrics['Sim_0.8_%s' % level_item] = gen_goal_08_level_list.count(level_item)
        metrics['Sim_0.8_train_0.7'] = gen_goal_train_08_number / gen_goal_08_num
        metrics['Sim_0.8_train_0.7_number'] = gen_goal_train_08_number
        for level_item in level_list:
            metrics['Sim_0.8_train_0.7_%s_number' % level_item] = gen_goal_08_train_07_level_list.count(level_item)
    if gen_goal_09_num > 0:
        metrics['Sim_0.9'] = gen_goal_09_num / len(gen)
        metrics['Sim_0.9_number'] = gen_goal_09_num
        for level_item in level_list:
            metrics['Sim_0.9_%s' % level_item] = gen_goal_09_level_list.count(level_item)
        metrics['Sim_0.9_train_0.7'] = gen_goal_train_09_number / gen_goal_09_num
        metrics['Sim_0.9_train_0.7_number'] = gen_goal_train_09_number
        for level_item in level_list:
            metrics['Sim_0.9_train_0.7_%s_number' % level_item] = gen_goal_09_train_07_level_list.count(level_item)
    table = pd.DataFrame([metrics]).T
    table.to_csv(metrics_out, header=False)

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main(config)
