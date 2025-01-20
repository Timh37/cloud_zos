import pandas as pd
import numpy as np
import intake
from xmip.utils import google_cmip_col

'''set of functions to generate and manipulate catalogues of CMIP6 data'''

def search_cloud(variable_id=None,experiment_id=None,table_id=None,require_all_on=None):
    
    kwargs = locals().copy()
    for k,v in locals().items():
        if v is None:
            kwargs.pop(k)

    def_col = google_cmip_col() #google cloud catalog
    qc_col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/cmip6-pgf-ingestion-test/catalog/catalog.json") #temporary pangeo-leap-forge catalogue
    nonqc_col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/cmip6-pgf-ingestion-test/catalog/catalog_noqc.json")
    #set from which catalogues data is preferred if duplicates are present (higher=higher priority)
    col_df = def_col.df
    qc_df = qc_col.df
    nonqc_df = nonqc_col.df
    
    col_df['prio'] = 1
    qc_df['prio'] = 2
    nonqc_df['prio'] = 0
    
    def_col.esmcat._df = pd.concat([col_df,qc_df,nonqc_df],ignore_index=True) #merge catalogs

    cat = def_col.search(**kwargs)
    cat = drop_older_versions(cat)
    return cat
    
def reduce_cat_to_max_num_realizations(cmip6_cat):
    '''Reduce grid labels in pangeo cmip6 catalogue by 
    keeping grid_label and 'ipf' identifier combination with most datasets (=most realizations if using require_all_on)'''
    df = cmip6_cat.df
    cols = df.columns.tolist()
    
    df['ipf'] = [s[s.find('i'):] for s in df.member_id] #extract 'ipf' from 'ripf'

    #generate list of tuples of (source_id,ipf,grid_label) that provide most realizations (note this will omit realizations not available at this grid but possibly at others)
    max_num_ds_tuples = df.groupby(['source_id','ipf'])['grid_label'].value_counts().groupby(level=0).head(1).index.to_list() #head(1) gives (first) max. value since value_counts sorts max to min
    df_filter = pd.DataFrame(max_num_ds_tuples,columns=['source_id','ipf','grid_label']) #generate df to merge catalogue df on
    
    df = df_filter.merge(right=df, on = ['source_id','ipf','grid_label'], how='left') #do the subsetting
    df = df.drop(columns=['ipf']) #clean up
    df= df[cols]

    cmip6_cat.esmcat._df = df #(columns now ordered differently, probably not an issue?)
    return cmip6_cat

def drop_vars_from_cat(cmip6_cat,vars_to_drop):
    '''drops entries with unwanted variables from search catalogue dataframe'''
    cmip6_cat.esmcat._df = cmip6_cat.df.drop(cmip6_cat.df[cmip6_cat.df.variable_id.isin(vars_to_drop)].index).reset_index(drop=True)
    return cmip6_cat

def drop_older_versions(cat):
    '''functionality to drop duplicate datasets if ingesting from multiple catalogues'''
    
    for i in np.arange(len(cat.df)): #check if version is an integer or a string prepended with 'v', and change all to integers
        if isinstance(cat.df.loc[i,'version'],int)==False:
            cat.df.loc[i,'version'] = int(cat.df.loc[i,'version'].replace('v',''))
            
    #sort by catalog priority for duplicate versions and drop duplicates (drop duplicates keeps the first dataset (=highest catalog priority))
    cat.esmcat._df = cat.df.sort_values(by='prio', ascending=False).drop_duplicates(subset=['activity_id','institution_id','source_id','experiment_id','member_id','table_id','variable_id','grid_label','version']).sort_index()
    
    #then sort by version and drop duplicates
    cat.esmcat._df = cat.df.sort_values(by='version', ascending=False).drop_duplicates(subset=['activity_id','institution_id','source_id','experiment_id','member_id','table_id','variable_id','grid_label']).sort_index()
    
    return cat

def reduce_areacello_cat(areacello_cat):
    #remove duplicate source/grids & PMIP runs with potentially different masks
    areacello_cat.esmcat._df = areacello_cat.df.drop_duplicates(subset=['source_id','grid_label'])[['activity_id',	'institution_id',	'source_id',	'experiment_id',	'member_id',	'table_id',	'variable_id',	'grid_label',	'zstore',	'dcpp_init_year','version']]
    areacello_cat.esmcat._df = areacello_cat.esmcat._df.drop(areacello_cat.esmcat._df[areacello_cat.esmcat._df.activity_id == 'PMIP'].index) #land mask may be different from contemporary experiments
    return areacello_cat