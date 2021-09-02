import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, stk_hld_csv):
        self.act_id = stk_hld_csv['act_id'].values
        self.item_cd = stk_hld_csv['iem_cd'].values
        self.hold_d = self.scale(stk_hld_csv['hold_d'].values)
        self.iem_info,self.cus_info = self.load_data()

    def __len__(self):
        return len(self.act_id)

    def __getitem__(self, idx):
        act = self.act_id[idx]
        item = self.item_cd[idx]
        hold_d = self.hold_d[idx]
        act_vec = self.cus_info[self.cus_info['act_id'] == act][['sex_dit_cd', 'cus_age_stn_cd', 'ivs_icn_cd','cus_aet_stn_cd', \
                                                                'mrz_pdt_tp_sgm_cd', 'lsg_sgm_cd', 'tco_cus_grd_cd','tot_ivs_te_sgm_cd', 'mrz_btp_dit_cd']].values
        item_vec = self.iem_info[self.iem_info['iem_cd']==item][['btp_cfc_cd', 'mkt_pr_tal_scl_tp_cd','stk_dit_cd']].values

        act_vec,item_vec = self.reshape(act_vec),self.reshape(item_vec) 
        return act_vec,item_vec,hold_d

    def load_data(self):
        iem_info = pd.read_csv('data/iem_info_20210902.csv')
        cus_info = pd.read_csv('data/cus_info.csv')
        return iem_info,cus_info
    
    def reshape(self,x):
        return torch.squeeze(torch.LongTensor(x))
    
    def scale(self,x):
        robustScaler = RobustScaler()
        return robustScaler.fit_transform(x.reshape(-1,1)).reshape(-1)
