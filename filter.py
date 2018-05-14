import json
import time
from datetime import datetime
import sys
import numpy as np
import pickle


protocol_number_dict = {
    'tcp': 6,
    'udp': 17
}


def avg(lst):
    return sum(lst)/len(lst)


def scale(dataset, trained=False):
    result = []
    maxims = []
    minims = []
    if not trained:
        n_features = len(dataset[0]) - 1
        for i in range(n_features):
            current_set = [x[i] for x in dataset]
            maxim = max(current_set)
            minim = min(current_set)
            minims.append(minim)
            maxims.append(maxim)
            req_set = [(x-minim)/(maxim-minim) for x in current_set]
            result.append(req_set)
        result.append([x[-1] for x in dataset])
        with open('meta.pkl', 'wb+') as mta:
            data = pickle.dumps([(mx, mn) for mx, mn in zip(maxims, minims)])
            mta.write(data)
        return np.transpose(result)
    else:
        data = []
        with open('meta.pkl', 'rb') as mta:
            data = pickle.loads(mta.read())
        for i in range(n_features):
            current_set = [x[i] for x in dataset]
            maxim = data[i][0]
            minim = data[i][1]
            minims.append(minim)
            maxims.append(maxim)
            req_set = [(x-minim)/(maxim-minim) for x in current_set]
            result.append(req_set)
        result.append([x[-1] for x in dataset])
        return np.transpose(result)


def meanify(dataset, tDiff):
    print (len(dataset))
    t_now = dataset[0][-1]
    meaned_dataset = []
    currs = []
    for d in dataset:
        if d[-1] - t_now > tDiff:
            meaned_now = []
            for i in range(0, len(d)):
                meaned_now.append(avg([c[i] for c in currs]))
            meaned_dataset.append(meaned_now)
            t_now = d[-1]
            currs = [d]
        else:
            currs.append(d)
    meaned_now = []
    for i in range(0, len(d)):
        meaned_now.append(avg([c[i] for c in currs]))
    meaned_dataset.append(meaned_now)
    return meaned_dataset


def generate_set(datum):
    try:
        src_bytes = datum['source']['stats']['net_bytes_total']
    except Exception:
        return None
    try:
        dest_bytes = datum['dest']['stats']['net_bytes_total']
    except Exception:
        return None
    tot_bytes = src_bytes + dest_bytes
    try:
        src_packets = datum['source']['stats']['net_packets_total']
    except Exception:
        return None
    try:
        dest_packets = datum['source']['stats']['net_packets_total']
    except Exception:
        return None
    tot_packets = src_packets + dest_packets
    protocol_number = protocol_number_dict[datum['transport']]
    try:
        byte_packet_ratio = tot_bytes/tot_packets
    except Exception:
        return None
    # 2018-04-18T06:23:23.032Z
    try:
        time_end = time.mktime(datetime.strptime(datum['last_time'],
                                                 '%Y-%m-%dT%H:%M:%S.%fZ')
                               .timetuple())
        time_start = time.mktime(datetime.strptime(datum['start_time'],
                                                   '%Y-%m-%dT%H:%M:%S.%fZ')
                                 .timetuple())
    except Exception:
        print (datetime.strptime(datum['last_time'], '%Y-%m-%dT%H:%M:%S.%fZ'))
        print (datetime.strptime(datum['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ'))
        sys.exit(0)
    time_diff = time_end - time_start
    time_avg = (time_end + time_start)/2
    try:
        byte_rate = tot_bytes/time_diff
    except ZeroDivisionError:
        byte_rate = tot_bytes
    this_list = [tot_bytes, byte_rate, byte_packet_ratio, time_avg]
    return this_list


def parse_json_data(jsonString):
    jData = json.loads(jsonString)
    is_list = False
    ret_list = []
    try:
        jData['source']
    except TypeError:
        is_list = True
    if is_list:
        for datum in jData:
            z = generate_set(datum)
            if z is not None:
                ret_list.append(z)
        ret_list = scale(meanify(ret_list, 30))
        # ret_list = meanify(scale(ret_list), 1)
        return ret_list
    else:
        return [generate_set(jData)]


def create_data(trained=False):
    if not trained:
        with open('packetbeats') as pkb:
            values = pkb.read()
    else:
        with open('packetbeats_test') as pkb:
            values = pkb.read()

    data = values.split('\n')
    data = ',\n'.join(data[:-1])
    data = parse_json_data('[' + data + ']')

    dicts = []
    for d in data:
        curr_dict = {
            'bytes_out': d[0],
            'byte_rate': d[1],
            'byte_packet_ratio': d[2],
            'time_avg': d[3]
        }
        if not trained:
            curr_dict['class'] = discriminant(d[2], d[1], d[0])
        dicts.append(curr_dict)

    with open('pb_train.json'
              if not trained else 'pb_classify.json', 'w') as pbs:
        pbs.write(json.dumps(dicts))

    return 0


def fuzzy_and(a, b):
    return min(a, b)


def fuzzy_or(a, b):
    return max(a, b)


def fuzzy_not(a):
    return 1-a


def fuzzy_xor(a, b):
    return fuzzy_or(fuzzy_and(fuzzy_not(a), b), fuzzy_and(a, fuzzy_not(b)))


def discriminant(a, b, c):
    return (round(fuzzy_or(fuzzy_not(c),
                  fuzzy_and(a, b))) + round(fuzzy_not(fuzzy_or(fuzzy_not(b),
                                                               c))))/2
