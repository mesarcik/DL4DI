import json
import logging
import collections 

import h5py
import sys
sys.path.append('../lofar_data_interface')

import lofarReadSnippet 
from stations import stations

logger = logging.getLogger(__name__)


class ObservationDao(object):
    @staticmethod
    def read_sap_numbers(path):
        return lofarReadSnippet.read_sap_numbers(path)

    @staticmethod
    def read_sap_targets(path):
        return lofarReadSnippet.read_SAP_targets(path)

    @staticmethod
    def get_observation(path, sap):
        print('_________________________________________________________________')
        print('this is path: \n {}'.format(path))
        print('this is sap: \n {}'.format(sap))
        logger.info('Loading observation for sap %d from: %s', sap, path)
        data = lofarReadSnippet.read_hypercube(path, visibilities_in_dB=True, read_visibilities=True,
                                               read_flagging=False)

        file_annotations = lofarReadSnippet.read_file_annotations(path)
        clusters, cluster_algo_annotations = lofarReadSnippet.read_clusters(path)
        info_string = lofarReadSnippet.create_info_string(data, path, file_annotations, clusters, cluster_algo_annotations)
        logger.info('Loaded observation for %s\n%s', path, info_string)
        return data['saps'][sap]

    @staticmethod
    def get_observation_id(path, sap=0,label=None):
        logger.info('Loading observation id from: %s', path)
        data = lofarReadSnippet.read_hypercube(path, read_visibilities=False, read_flagging=False)

        return data.get('sas_id', 'unknown')

    @staticmethod
    def get_cluster_annotations(path, sap, label='latest'):
        logger.info('Loading cluster annotations for label="%s" sap %d from: %s', label, int(sap), path)
        data = lofarReadSnippet.read_hypercube(path, read_visibilities=False, read_flagging=False)
        file_annotations = lofarReadSnippet.read_file_annotations(path)
        clusters, cluster_algo_annotations = lofarReadSnippet.read_clusters(path)
        info_string = lofarReadSnippet.create_info_string(data, path, file_annotations, clusters, cluster_algo_annotations)
        logger.info('Loaded cluster annotations for: %s\n%s', path, info_string)

        return clusters[int(sap)].get('annotations', {})

    @staticmethod
    def get_clusters(path, sap, label):
        logger.info('Loading clusters from: %s', path)
        result = lofarReadSnippet.read_clusters(path, label)
        if result[0] == {}:
            # No saps means no clusters either.
            return {}
        return result[0][sap]['clusters']

    @staticmethod
    def get_annotations(path, sap, label):
        logger.info('Loading annotations from: %s', path)
        result = lofarReadSnippet.read_clusters(path, label)
        if result[0] == {}:
            # No saps means no clusters either.
            return {}
        return result[0][sap]['annotations']

    @staticmethod
    def get_parset(path):
        logger.info('Loading parset from: %s', path)
        return lofarReadSnippet.read_hypercube_parset(path, as_string=True)

    @staticmethod
    def get_info_dict(path):
        logger.info('Loading info_dict from: %s', path)
        return lofarReadSnippet.read_info_dict(path)

    @staticmethod
    def write_clusters(path, sap, clusters, label):
        logger.info('Writing clusters to: %s', path)
        lofarReadSnippet.write_clusters(path, {sap: {'clusters': clusters}}, label=label)

    @staticmethod
    def annotate_cluster(cluster, path, annotation, user, sap, label='latest'):
        lofarReadSnippet.annotate_cluster(path, label, sap, cluster, annotation, user)

    @staticmethod
    def delete_cluster_annotation(path, sap_nr, cluster_nr, annotation_nr, label='latest'):
        lofarReadSnippet.delete_cluster_annotation(path, sap_nr, cluster_nr, annotation_nr, label='latest')

    @staticmethod
    def save_embedding(path, sap, embedding, label):
        ObservationDao._save_dataset_as_json(embedding, 'embedding', label, path, sap)

    @staticmethod
    def get_embedding(path, sap, label='latest'):
        return ObservationDao._get_dataset('embedding', label, path, sap)

    @staticmethod
    def save_labels(path, sap, labels, label):
        ObservationDao._save_dataset_as_json(labels, 'labels', label, path, sap)

    @staticmethod
    def get_labels(path, sap, label='latest'):
        return ObservationDao._get_dataset('labels', label, path, sap)


    @staticmethod
    def get_stations(path, sap, label='latest'):
        return ObservationDao._get_dataset('stations', label, path, sap)

    @staticmethod
    def get_stations_old(path, sap, label='latest'):
        return stations

    @staticmethod
    def save_stations(path, sap, stations, label):
        ObservationDao._save_dataset_as_json(stations, 'stations', label, path, sap)

    @staticmethod
    def save_baselines(path, sap, baselines, label):
        """
        Save baselines information into observation file.
        :param path: Observation file path
        :param sap: Sap number of which the relevant station information will be saved
        :param baselines: List of baselines
        :param label: Job label
        :return:
        """
        bak = baselines
        baseline_counter = 0
        for baseline in baselines:
            temp_list = [] 
            for b in baseline:
                if isinstance(b, bytes):
                    temp_list.append(b.decode("utf-8"))
                else:
                    break 
            bak[baseline_counter] = (temp_list[0],temp_list[1]) 
            baseline_counter+=1
        baselines =bak
        ObservationDao._save_dataset_as_json(baselines, 'baselines', label, path, sap)

    @staticmethod
    def get_baselines(path, sap, label='latest'):
        return ObservationDao._get_dataset('baselines', label, path, sap)

    @staticmethod
    def get_models(path, sap, label='latest'):
        data = (h5py.File(path)['clustering'].keys())
        return json.loads(json.dumps((data)))

    @staticmethod
    def get_performance(path, sap,label):
        print('_________________________________________________________________')
        print('_________________________________________________________________')
        print('_________________________________________________________________')
        print('_________________________________________________________________')
        print(label , (h5py.File(path)['clustering/{}/saps/0/score'.format(label)][()]))
        print('_________________________________________________________________')
        print('_________________________________________________________________')
        print('_________________________________________________________________')
        print('_________________________________________________________________')
        data = (h5py.File(path)['clustering/{}/saps/0/score'.format(label)][()])
        return json.loads(json.dumps((data)))

    @staticmethod
    def _convert(data):
        if isinstance(data, basestring):
            return str(data)
        elif isinstance(data, collections.Mapping):
            return dict(map(convert, data.iteritems()))
        elif isinstance(data, collections.Iterable):
            return type(data)(map(ObservationDao._convert, data))
        else:
            return data

    @staticmethod
    def _get_dataset(dataset_name, label, path, sap):
        with h5py.File(path, "r+") as observation:
            h5path = ['clustering', label, 'saps', str(sap), 'json', dataset_name, 0]
            dataset = _navigate_to_h5path(observation, h5path, path)
            return json.loads(dataset)


    @staticmethod
    def _save_dataset_as_json(dataset, dataset_name, label, path, sap):
        encoded = json.dumps(dataset).encode(encoding='ascii')
        with h5py.File(path, "r+") as observation:
            h5path = ['clustering', label, 'saps', str(sap)]
            sap_group = _navigate_to_h5path(observation, h5path, path)
            if 'json' not in sap_group:
                json_group = sap_group.create_group('json')
            else:
                json_group = sap_group['json']
            json_group.create_dataset(dataset_name, (1,), h5py.special_dtype(vlen=str), encoded)


def _navigate_to_h5path(h5group, h5path, h5file='h5 file'):
    print(h5path)
    """
    Navigate to a path within a h5 group. Raises a more readable exception if key not found than standard key error.
    :param h5group:
    :param h5path: a list of keys composing the path for example ['path', 'to', 'item', '0']
    :param h5file:
    :return:
    """
    current_group = h5group
    current_path = []
    for key in h5path:
        try:
            current_group = current_group[key]
        except KeyError as e:
            raise KeyError(
                'No "{}" in path "{}" while navigating {}.'.format(key, '/' + '/'.join(current_path), h5file), e)
        current_path.append(key)
    return current_group
