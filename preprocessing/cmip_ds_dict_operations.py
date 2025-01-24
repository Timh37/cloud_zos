import xarray as xr
import numpy as np
from collections import defaultdict
from tqdm.autonotebook import tqdm
import xesmf as xe
import cftime
import os
import gcsfs
import dask
fs = gcsfs.GCSFileSystem()
'''set of functions to generate and manipulate dictionaries of CMIP6 datasets'''

def generate_dict_of_datasets(cat,models_to_exclude,preprocessing_func):
    cat.esmcat._df = cat.df[['activity_id',	'institution_id',	'source_id',	'experiment_id',	'member_id',	'table_id',	'variable_id',	'grid_label',	'zstore',	'dcpp_init_year','version']]
    for model in models_to_exclude: #preprocessing issues, regarding time coordinate (KIOST) and vertices?
        cat.esmcat._df = cat.esmcat._df.drop(cat.esmcat._df[(cat.esmcat._df.source_id == model)].index)
    cat.esmcat.aggregation_control.groupby_attrs = [] 
    ddict = cat.to_dataset_dict(**{'zarr_kwargs':{'consolidated':True,'use_cftime':True},'aggregate':True,'preprocess':preprocessing_func}) 
    return ddict
    
def select_period(ddict_in,start_year,end_year):
    '''select time period from datasets in dictionary'''
    assert start_year<end_year
    ddict_out = defaultdict(dict) #initialize output
    
    for k, v in ddict_in.items(): # for each entry (key, dataset)
        ddict_out[k] = v.sel(time=slice(str(start_year), str(end_year)))
    return ddict_out

def _match_twosided_attrs(ds_a, ds_b, attrs_a, attrs_b): #custom version of _match_attrs in xmip that allows to compare differently named attributes between datasets
    """returns the number of matched attrs between two datasets"""
    if len(attrs_a)!=len(attrs_b):
        raise Exception('lists of attributes in each dataset must be of equal length.')
        
    try:
        n_match = sum([ds_a.attrs[attrs_a[i]] == ds_b.attrs[attrs_b[i]] for i in range(len(attrs_a))])
        return n_match
    except:
        return 0

def fix_inconsistent_calendars(ds_ddict):
        for k,ds in ds_ddict.items():
            try:
                ds.time[-1] - ds.time[0]
            except: #unify calendars 
                not_prolgreg = np.where(np.array([type(i) for i in ds.time.values]) != cftime._cftime.DatetimeProlepticGregorian)[0] #find where calendar is not proleptic gregorian
                converted_time = ds.isel(time=not_prolgreg).convert_calendar('proleptic_gregorian',use_cftime=True).time #convert at these indices
                newtime = ds.time.values #replace old time index with new values
                newtime[not_prolgreg] = converted_time.values
                ds_ddict[k]['time'] = newtime
            
        return ds_ddict
    
def find_matching_pic_datasets(ds_ddict,pic_ddict,variable_id,min_numYrs_pic):
    datasets_without_pic = []

    attrs_a = ['parent_source_id','grid_label','parent_variant_label']
    attrs_b = ['source_id','grid_label','variant_label']

    for i,ds in tqdm(ds_ddict.items()):
        ## adapted from '_match_datasets' (which currently does not take differently named attributes to be matched:)
        matched_datasets = []
        pic_keys = list(pic_ddict.keys())
        for k in pic_keys:
            if _match_twosided_attrs(ds, pic_ddict[k], attrs_a,attrs_b) == len(attrs_a): #
                if len(np.unique(pic_ddict[k].time.dt.year))>=min_numYrs_pic: #length requirement piControl
                    if (pic_ddict[k].time[1]-pic_ddict[k].time[0]).dtype != (ds.time[1]-ds.time[0]).dtype: #check if deltatime units are equal between pic and experiment to be dedrifted, otherwise polyval on .time goes wrong? not with units m/month
                        print('Time units matched piControl dataset different from dataset to be dedrifted, not matching piControl dataset to: '+i)
                        continue
                    else:
                        ds.attrs['matched_pic_ds'] = k
                    
        if 'matched_pic_ds' not in ds:
            datasets_without_pic.append(i)
    return ds_ddict, datasets_without_pic

def store_matched_pic_linfit(ds_ddict,pic_ddict,variable_id,out_path):

    for k,ds in tqdm(ds_ddict.items()):
        if 'matched_pic_ds' in ds.attrs:
            pic_ds = pic_ddict[ds.attrs['matched_pic_ds']]

            fn = '.'.join(ds.attrs['matched_pic_ds'].split('.')[0:8])

            output_fn = os.path.join(out_path,fn) 
            if fs.exists(output_fn) == False:
                pic_ds['time'] = np.arange(0,len(pic_ds.time)) #replace time with simple arange, to get trend in m/month (assumption that every month is the same length, which is not entirely true, but ok)
                pic_fit = pic_ds[variable_id].polyfit(dim='time',deg=1) #linear fit in units m/month
                pic_fit = pic_fit.chunk({'x':ds[variable_id].chunksizes['x'],'y':ds[variable_id].chunksizes['y']})
                pic_fit.attrs['slope_units'] = 'm/month'
                
                pic_fit.to_zarr(output_fn,mode='w')
                
                pic_ds.close()
                pic_fit.close()
    return ds_ddict

def subtract_pic_linfit(ds_ddict,variable_id,in_path):
    dedrifted_ddict = defaultdict(dict)
    
    for k,ds in tqdm(ds_ddict.items()):
        if 'matched_pic_ds' in ds.attrs:
            fn = '.'.join(ds.attrs['matched_pic_ds'].split('.')[0:8])
            
            input_fn = os.path.join(in_path,fn)
            
            pic_fit = xr.open_dataset(input_fn,engine='zarr') #load picfit

            #adapted from xmip:
            chunks = ds[variable_id].isel({di: 0 for di in ds[variable_id].dims if di != 'time'}).chunks
            trend_time = dask.array.arange(0, len(ds.time), chunks=chunks, dtype=ds[variable_id].dtype)
    
            trend_time_da = xr.DataArray(
                trend_time,
                dims=['time'],
            )
            
            trend = (pic_fit.polyfit_coefficients.sel(degree=1).isel(member_id=0,drop=True) * trend_time_da)
            
            ds[variable_id] = ds[variable_id]-trend
    
            
            dedrifted_ddict[k] = ds
            pic_fit.close()
            
    return dedrifted_ddict
''' old
def subtract_pic_linfit(ds_ddict,variable_id,in_path):

    dedrifted_ddict = defaultdict(dict)
    
    for k,ds in tqdm(ds_ddict.items()):
        if 'matched_pic_ds' in ds.attrs:
            print(k)
            fn = '.'.join(ds.attrs['matched_pic_ds'].split('.')[0:8])
            
            input_fn = os.path.join(in_path,fn)
            
            pic_fit = xr.open_dataset(input_fn,engine='zarr') #load picfit
            
            ds_drift = xr.polyval(ds.time,pic_fit.sel(degree=1)) #evaluate fit
            ds_drift = ds_drift - ds_drift.isel(time=0) #remove intercept
           
            dedrifted_ddict[k] = ds
            dedrifted_ddict[k][variable_id] = ds[variable_id] - ds_drift.polyfit_coefficients.isel(member_id=0,drop=True) #drop parent member_id because it may differ from the ds member_id
            ds_drift.close()
            pic_fit.close()
            
    return dedrifted_ddict
'''    
def dedrift_datasets_linearly(ds_ddict,pic_ddict,variable_id,min_numYrs_pic):
    #note: assumed both dataset dicts need to have the same frequency!
    #note: this is different from the default xmip dedrifting because it computes the linear drift over the full piControl length instead of only the part overlapping with the experiment. The reason is that sometimes the piControl simulation is too short to cover all experiments.
    ds_ddict_dedrifted = defaultdict(dict)
    drift_ddict = defaultdict(dict)
    
    datasets_without_pic = []

    attrs_a = ['parent_source_id','grid_label','parent_variant_label']
    attrs_b = ['source_id','grid_label','variant_label']

    for i,ds in tqdm(ds_ddict.items()):
        #_match_datasets would ideally be used for this, but currently does not take differently named attributes to be matched:

        ## adapted from '_match_datasets'
        matched_datasets = []
        pic_keys = list(pic_ddict.keys())
        for k in pic_keys:
            if _match_twosided_attrs(ds, pic_ddict[k], attrs_a,attrs_b) == len(attrs_a): #
                if len(np.unique(pic_ddict[k].time.dt.year))>=min_numYrs_pic: #length requirement piControl
                    ds_matched = pic_ddict[k]
                    # preserve the original dictionary key of the chosen dataset in attribute.
                    ds_matched.attrs["original_key"] = k
                    matched_datasets.append(ds_matched) #if multiple, we just take the first one for now..
                    
        if len(matched_datasets) == 0:
            datasets_without_pic.append(i)
            #print('No (long enough) piControl found for: '+i)
        else: #do the dedrifting
            pic_ds = matched_datasets[0] #take first matching dataset
            pic_fit = pic_ds[variable_id].polyfit(dim='time',deg=1) #linear fit

            if (pic_ds.time[1]-pic_ds.time[0]).dtype != (ds.time[1]-ds.time[0]).dtype: #check if deltatime units are equal between pic and experiment to be dedrifted
                print('Time units piControl dataset different from dataset to be dedrifted, cannot dedrift: '+i)
                datasets_without_pic.append(i) #if there are more matching datasets, then these could also be checked (but in this workflow this doesn't happen?)
                continue
            else:    
                ds_drift = xr.polyval(ds.time,pic_fit) #evaluate fit
                ds_drift = ds_drift - ds_drift.isel(time=0) #remove intercept

                drift_ddict[i] = ds_drift
                
                ds_ddict_dedrifted[i] = ds
                ds_ddict_dedrifted[i][variable_id] = ds[variable_id] - ds_drift.polyfit_coefficients.isel(member_id=0,drop=True) #drop parent member_id because it may differ from the ds member_id
                ds_ddict_dedrifted[i].attrs['dedrifted_with'] = pic_ds.attrs['original_key']
                
    return ds_ddict_dedrifted, drift_ddict, datasets_without_pic

# create regridders per source_id (datasets of source_id should be on the same grid after passing through reduce_cat_to_max_num_realizations
def create_regridder_dict(dict_of_ddicts, target_grid_ds):

    regridders = {}
    source_ids = np.unique(np.hstack([[ds.attrs['source_id'] for ds in dataset_dict.values()] for dataset_dict in dict_of_ddicts.values()]))
    
    for si in tqdm(source_ids):
        for ds_ddict in dict_of_ddicts.values():
            matching_keys = [k for k in ds_ddict.keys() if k.split('.')[0]==si]
            if len(matching_keys)>0:
                # take the first one (we don't really care here which one we use)
                ds = ds_ddict[matching_keys[0]]
                regridders[si] = xe.Regridder(ds,target_grid_ds,'bilinear',ignore_degenerate=True,periodic=True,unmapped_to_nan=True) #create regridder for this source_id
                continue
    return regridders

def regrid_datasets_in_ddict(ds_ddict,regridder_dict):
    for key,ds in tqdm(ds_ddict.items()):
        regridder = regridder_dict[ds.attrs['source_id']] #select regridder for this source_id
        regridded_ds = regridder(ds, keep_attrs=True) #do the regridding
        
        ds_ddict[key] = regridded_ds
    return ds_ddict
            
def subtract_ocean_awMean(ds_ddict,variable_id):
    noMean_ddict = defaultdict(dict)
    for k,v in tqdm(ds_ddict.items()):
        if 'areacello' not in v:
            print('Could not find "areacello" in dataset: '+k)
            continue
        else:
            noMean_ddict[k] = v
            noMean_ddict[k][variable_id] = noMean_ddict[k][variable_id] - noMean_ddict[k][variable_id].weighted(noMean_ddict[k].areacello.fillna(0)).mean(['x','y'],skipna=True)
    return noMean_ddict
    
def pr_flux_to_m(ddict_in):
    '''convert pr flux to total accumulated pr'''
    ddict_out = ddict_in
    for k, v in ddict_in.items(): # for each entry (key, dataset)
        if 'pr' in v.variables: #if dataset contains pr
            assert v.pr.units == 'kg m-2 s-1' #check if units are flux

            with xr.set_options(keep_attrs=True): #convert 'kg m-2 s-1' to daily accumulated 'm'
                v['pr'] = 24*3600*v['pr']/1000 #multiply by number of seconds in a day to get to kg m-2, and divide by density (kg/m3) to get to m    
            v.pr.attrs['units'] = 'm'
        ddict_out[k] = v
    return ddict_out

def drop_duplicate_timesteps(ddict_in):
    '''removes duplicate timesteps in CMIP6 simulations in dictionary if present'''
    ddict_out = defaultdict(dict)
    for k, v in ddict_in.items(): # for each entry (key, dataset)
        if 'time' in v:
            try:
                unique_time, idx = np.unique(v.time,return_index=True)
            except:
                print('Could not determine unique timesteps in: '+k+', dropping dataset.') #some time coordinates contain floats in parts of the experiment? to-do: have a closer look
                continue
            if len(v.time) != len(unique_time):
                v = v.isel(time=idx)
                print('Dropping duplicate timesteps for:' + k)  
            ddict_out[k] = v
        else:
            ddict_out[k] = v
            
    return ddict_out

def drop_coords(ddict_in,coords_to_drop):
    '''remove unwanted coordinates from datasets in dictionary'''
    for k, v in ddict_in.items(): # for each entry (key, dataset)  
        ddict_in[k] = v.drop_dims(coords_to_drop,errors="ignore")
    return ddict_in

def drop_vars(ddict_in,vars_to_drop):
    '''remove unwanted variables from datasets in dictionary'''
    for k, v in ddict_in.items(): # for each entry (key, dataset)  
        ddict_in[k] = v.drop_vars(vars_to_drop,errors="ignore")
    return ddict_in

def drop_incomplete(ddict_in):
    ''' drop datasets with timesteps that are not monotonically increasing or that have large gaps in time'''
    ddict_out = defaultdict(dict)
    
    for k, ds in ddict_in.items(): # for each entry (key, dataset)  
        
        time_diff = ds.time.diff('time').astype(int)
        mean_time_diff = time_diff.mean()
        normalized_time_diff = abs((time_diff - mean_time_diff)/mean_time_diff)
        
        # do not include datasets with time not increasing monotonically
        if (time_diff > 0).all() == False: 
            continue
        
        # do not include datasets with large time gaps
        if (normalized_time_diff>0.05).all():
            continue
        
        ddict_out[k] = ds
        
    return ddict_out

def get_availability_from_ddicts(dict_of_ddicts):
    all_models = np.unique(np.hstack([[ds.attrs['source_id'] for ds in dataset_dict.values()] for dataset_dict in dict_of_ddicts.values()]))
    availability = defaultdict(dict)
    
    for k,ddict in dict_of_ddicts.items():
        model_members = defaultdict(dict)
        
        for model in all_models:
            members = np.unique([ds.member_id for ds in ddict.values() if ds.source_id == model])
            model_members[str(model)] = list(members)
            
        availability[k] = model_members
    
    availability['all'] = defaultdict(dict)
    for model in all_models:
        model_member_list = []
        for k in dict_of_ddicts.keys():
            model_member_list.append(availability[k][model])
        availability['all'][str(model)] = list(set.intersection(*map(set,model_member_list)))
    return availability