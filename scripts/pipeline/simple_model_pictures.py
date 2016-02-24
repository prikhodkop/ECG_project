#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
import pandas as pd
import numpy as np

import scipy.stats as sta

import time

import sys
sys.path.append('..')

try:
  import user_project_config as conf
except:
  import project_config as conf

from IO import data_loading as dl

import matplotlib.pyplot as plt

print 'Initialization'
t0 = time.time()

mortality = dl.read_dta('mortality_SAHR_ver101214', data_folder=conf.path_to_dta),
selected_pp =  dl.read_dta('selected_pp', data_folder=conf.path_to_dta)
sleep = dl.get_sleep_time(conf.path_to_dta, version=2)

data = pd.merge(selected_pp, sleep, on='GIDN')

GIDNs = sleep['GIDN']

def accumu(lis):
    total = 0
    for x in lis:
        total += x
        yield total

def get_sleep_state(times, start, end):
    for time in times:
        if time <= start:
            state = 0
        if  start <= time <= end:
            state = 1
        if time >= end:
            state = 2
        yield state

def create_subsample(data, window=200):
    idx = 0
    result = []
    while idx < len(data)-len(data)/100:
        result.append(np.median(data[idx:np.median((idx+window, len(data)))]))
        # result.append(np.median(data[idx:len(data)]))

        idx += window
    return result

def get_human_time(ms):
    x = int(round(ms / 1000))
    seconds = x % 60
    x /= 60
    minutes = x % 60
    x /= 60
    hours = x % 24
    x /= 24
    days = x
    return str(days)+'d'+str(hours)+'h'+str(minutes) + 'm'

filter_type = None # 'NN_only'
window_size = 20

start_from = 5
end_before = -5


print 'Process RR'

for i, gidn in enumerate(GIDNs):

    t0 = time.time()

    data_RR = dl.load_RR_data(gidn, path=conf.path_to_RR, version=2)

    if data_RR is None:
        continue

    data_RR['is_sleeping'] = 1*(data_RR['time'] >= data.iloc[i]['sleep_trend_start_corr']) + 1*(data_RR['time'] > data.iloc[i]['sleep_trend_end_corr'])
    data_RR['is_sleeping_diary'] = 1*(data_RR['time'] >= data.iloc[i]['sleep_diary_start_corr']) + 1*(data_RR['time'] > data.iloc[i]['sleep_diary_end_corr'])

    if filter_type is None:
        intervals = data_RR['interval'][:].as_matrix()
        sleeps = data_RR['is_sleeping'][:].as_matrix()
        sleeps_diary = data_RR['is_sleeping_diary'][:].as_matrix()
    elif filter_type == 'NN_only':
        condition = (data_RR['interval_type'] == 'NN') & (data_RR['interval'] > 300) & (data_RR['interval'] < 2000)
        intervals = data_RR[condition]['interval'][:].as_matrix()
        sleeps = data_RR[condition]['is_sleeping'][:].as_matrix()
        sleeps_diary = data_RR[condition]['is_sleeping_diary'][:].as_matrix()

    final_intervals = create_subsample(intervals, window=window_size)
    final_sleeps = create_subsample(sleeps, window=window_size)
    final_sleeps = np.array([int(round(sleep)) for sleep in final_sleeps])
    final_sleeps_diary = create_subsample(sleeps_diary, window=window_size)
    final_sleeps_diary = np.array([int(round(sleep)) for sleep in final_sleeps_diary])

    interval_time = np.array([elem for elem in accumu(final_intervals)])
    interval_time = (interval_time - np.min(interval_time))/(np.max(interval_time) - np.min(interval_time)) + 0.001
    interval_index = np.array(range(len(interval_time)))
    interval_index = (interval_index - np.min(interval_index))/(float(np.max(interval_index)) - np.min(interval_index)) + 0.001
    interval_sleep = final_sleeps
    interval_sleep_diary = final_sleeps_diary


    # filt_condition = interval_index > 0.1
    # interval_time = interval_time[filt_condition]
    # interval_index = interval_index[filt_condition]
    # interval_sleep = interval_sleep[filt_condition]
    # interval_sleep_diary = interval_sleep_diary[filt_condition]

    # print interval_time

    # print data.iloc[i]

    t1 = time.time()

    print 'Data perparation: ', t1 - t0

    slope2, intercept2, r_value, p_value, std_err = sta.linregress((interval_index), (interval_time))
    print 'before', slope2, intercept2

    signal_with_not_trend2 = (interval_time) - slope2*(interval_index) - intercept2

    grad_signal_with_not_trend2 = np.gradient(signal_with_not_trend2)
    # grad_signal_with_not_trend2 = np.gradient(signal_with_not_trend2)

    zero_grad_min_idxs =  np.abs(grad_signal_with_not_trend2) < 10.**-6
    zero_grad_max_idxs =  np.abs(grad_signal_with_not_trend2) < 10.**-6
    zero_grad_min_idxs[1:-1] =  (grad_signal_with_not_trend2[:-2]<=0)&(grad_signal_with_not_trend2[1:-1]>=0)
    zero_grad_max_idxs[1:-1] =  (grad_signal_with_not_trend2[:-2]>=0)&(grad_signal_with_not_trend2[1:-1]<=0)


    print 'Number of zero grads ', np.sum(zero_grad_max_idxs) + np.sum(zero_grad_min_idxs)

    t2 = time.time()
    print 'Signal processing: ', t2 - t1

    fig = plt.figure()
    ax = fig.add_subplot(411)
    ax.plot(interval_index[interval_sleep == 0], interval_time[interval_sleep == 0], c='b')
    ax.plot(interval_index[interval_sleep == 1], interval_time[interval_sleep == 1], c='b')
    ax.plot(interval_index[interval_sleep == 1], interval_time[interval_sleep == 1]-0.01, c='r')
    ax.plot(interval_index[interval_sleep_diary == 1], interval_time[interval_sleep_diary == 1]+0.01, c='g')
    ax.plot(interval_index[interval_sleep == 2], interval_time[interval_sleep == 2], c='b')

    ax.scatter(interval_index[zero_grad_min_idxs], interval_time[zero_grad_min_idxs], c='black', marker=">")
    ax.scatter(interval_index[zero_grad_max_idxs], interval_time[zero_grad_max_idxs], c='black', marker="<")


    ax.set_xlim([0.,1.2])

    if np.isnan(data.iloc[i]['sleep_trend_start_corr']) or np.isnan(data.iloc[i]['sleep_trend_end_corr']):
        continue
    start_time = get_human_time(data.iloc[i]['sleep_trend_start_corr'])
    end_time = get_human_time(data.iloc[i]['sleep_trend_end_corr'])
    duration = get_human_time(data.iloc[i]['sleep_trend_end_corr'] - data.iloc[i]['sleep_trend_start_corr'])
    # ax.set_title('GIDN: '+str(gidn)+'\n Start: '+str(start_time)+' End: '+str(end_time)+'\n Duration: '+str(duration))
    try:
        age = str(data.iloc[i]['Age'])
    except:
        age = 'nan'
    try:
        sex = str(data.iloc[i]['Sex_x'])
    except:
        sex = 'nan'
    try:
        sh = data.iloc[i]['SelfHealth2'].encode('utf-8')
        ax.set_title(sh)
    except:
        sh = 'nan'
    try:
        bmi = str(data.iloc[i]['BMIgr'])
    except:
        bmi = 'nan'

    duration_of_sleep_trend = np.sum(interval_sleep == 1)
    duration_of_sleep_diary = np.sum(interval_sleep_diary == 1)
    duration_of_sleep_avg = (duration_of_sleep_trend + duration_of_sleep_diary)/2

    idxs_open = np.array(range(len(zero_grad_min_idxs)))[zero_grad_min_idxs]
    idxs_close = np.array(range(len(zero_grad_max_idxs)))[zero_grad_max_idxs]

    durations_of_sleep_est = []
    for idx_open in idxs_open:
        for idx_close in idxs_close:
            if idx_close < idx_open:
                continue
            else:
                durations_of_sleep_est.append(idx_close - idx_open)
                break
    duration_of_sleep_est = max(durations_of_sleep_est)

    interval = int(round(np.median(intervals)))
    # print interval

    err_trend = np.abs(duration_of_sleep_trend - duration_of_sleep_est)*window_size*interval/60/1000
    err_diary = np.abs(duration_of_sleep_diary - duration_of_sleep_est)*window_size*interval/60/1000
    err_avg = np.abs(duration_of_sleep_avg - duration_of_sleep_est)*window_size*interval/60/1000

    duration2 = np.abs(duration_of_sleep_trend)*window_size*interval/60/1000/60.

    ax.set_title('GIDN: '+str(gidn)+'\n err_trend: '+str(err_trend)+' err_diary: '+str(err_diary)+' err_avg: '+str(err_avg)+' sex: '+sex+' Age: '+age+'\n Duration: '+str(duration)+' Duration2: '+str(duration2))

    # ax = fig.add_subplot(312)
    #
    # slope1, intercept1, r_value, p_value, std_err = sta.linregress((interval_index[interval_sleep == 0][1000:2000]), (interval_time[interval_sleep == 0][1000:2000]))
    # print 'before', slope1, intercept1
    # #
    # # slope, intercept, r_value, p_value, std_err = sta.linregress(np.log(interval_index[interval_sleep == 1]), np.log(interval_time[interval_sleep == 1]))
    # # print 'in', slope, intercept
    # #
    # # slope, intercept, r_value, p_value, std_err = sta.linregress(np.log(interval_index[interval_sleep == 2]), np.log(interval_time[interval_sleep == 2]))
    # # print 'after', slope, intercept
    #
    # signal_with_not_trend1 = (interval_time) - slope1*(interval_index) - intercept1
    #
    # ax.plot(interval_index, signal_with_not_trend1, c='b')

    ax = fig.add_subplot(412)


    ax.plot(interval_index, signal_with_not_trend2, c='b')

    ax = fig.add_subplot(413)


    # grad_grad_signal_with_not_trend2 = np.gradient(grad_signal_with_not_trend2)

    ax.plot(interval_index, np.abs(grad_signal_with_not_trend2), c='b')

    # poly = np.poly1d(np.polyfit((interval_index[interval_sleep == 0]), (interval_time[interval_sleep == 0]), 2))
    # ax.plot(interval_index, (interval_time) - poly(interval_index), c='b')

    ax = fig.add_subplot(414)
    ax.plot(interval_index, final_intervals/np.mean(final_intervals) - 1, c='b')

    # plt.show()
    fig.savefig('/Users/pavel.prikhodko/Desktop/ECG/ECG_project/simple_model/'+str(gidn)+'.png', dpi=fig.dpi)
    plt.close("all")

    t3 = time.time()
    print 'Drawing plots: ', t3 - t2

    print 'Total elapsed time (', i,')', time.time() - t0


    # if i > 10:
    #     break
