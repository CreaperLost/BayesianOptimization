import json
import os

import requests
import pandas
import tempfile

from jadbio_internal.api.auth import json_headers
from jadbio_internal.api.job import Job

from requests_toolbelt.multipart.encoder import MultipartEncoder

from jadbio_internal.api.task import wait_for_job

NEW_IMAGE_ENDPOINT = '/api/image_data_array/new'
ADD_IMAGE_ENDPOINT = '/api/image_data_array/add_image'
COMMIT_IMAGE_ENDPOINT = '/api/image_data_array/commit'
SUPPORTED_IMAGE_TYPES = ['jpeg', 'jpg', 'tif', 'png', 'bmp']


def json_encode_image_init(project_id: int, name: str, description: str, has_feature_names: bool):
    return json.dumps(
        {
            'projectId': project_id,
            'name': name,
            'description': description,
            'hasFeatureNames': has_feature_names
        }
    )


def find_sample_in_folder(folder: str, sample: str) -> str:
    for file in os.listdir(folder):
        split = file.split(".")
        if len(split) != 2:
            raise
        prefix = split[0]
        suffix = split[1]
        if prefix==sample and suffix in SUPPORTED_IMAGE_TYPES:
            return folder+'/'+file
    raise

def check_sample_names(sample_names: list):
    sample_count = {}
    for sname in sample_names:
        sample_count[sname] = sample_count.get(sname, 0)+1
        if(sample_count[sname])>1:
            raise('Sample {} is contained twice'.format(sname))

def write_target_dict_to_tmp(dict: dict):
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, 'w') as tmp:
        tmp.write('header,target\n')
        for (sample_key, target) in dict.items():
            tmp.write(f'{sample_key},{target}\n')
    return path

def image_files_in_dir(dir: str):
    return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.split(".")[1] in SUPPORTED_IMAGE_TYPES]

class ImageClient:
    host: str = None
    token: str = None

    def __init__(self, host: str, token: str):
        self.host = host
        self.token = token

    def upload(
            self,
            project: int,
            name: str,
            data_folder: str,
            target_path: str,
            has_feature_names = True,
            description: str = None
    ):
        """

        :param project: project id
        :param name: image name (should be unque per project)
        :param data_folder: folder containing all
        :param target_path: csv file containing sample names, target, other features, if None it defaults to 'target.csv'
        :param has_feature_names:
        :param description: dataset description
        :return: dataset id
        """

        target_path = "target.csv" if target_path is None else target_path
        job = self.__init_job(project, name, data_folder+target_path, has_feature_names, description)
        sample_names = list(pandas.read_csv(data_folder + 'target.csv').iloc[:, 0])
        check_sample_names(sample_names)
        for sample in sample_names:
            print('Adding sample {}'.format(sample))
            sample_path = find_sample_in_folder(data_folder, sample)
            self.__add_image(job, sample, sample_path)
        self.__commit(job)
        return wait_for_job(self.host, self.token, job)


    def upload_type2(
            self,
            project: int,
            name: str,
            data_folder: str,
            description: str = None
    ):
        """

        :param project: project id
        :param name: image name (should be unque per project)
        :param data_folder: folder that contains images in the following format:
        folder
              -> class1
                        -> img1
                        -> img2
                        ...
              -> class2
                        -> img3
                        -> img4
                        ...
              ...
        :param description: dataset description
        :return: dataset id
        """

        class_names = os.listdir(data_folder)
        if(len(class_names)<2):
            raise(f'Classes found {len(class_names)}')

        class_dict = {}
        sample_path_dict = {}
        for target_class in class_names:
            image_files = image_files_in_dir(os.path.join(data_folder, target_class))

            for image_file in image_files:
                sample = image_file.split('.')[0]
                class_dict[sample] = target_class
                sample_path_dict[sample] = os.path.join(data_folder, target_class, image_file)

        try:
            target_path = write_target_dict_to_tmp(class_dict)
            with open(target_path, "r") as myfile:
                print(myfile.readlines())
            job = self.__init_job(project, name, target_path, True, description)
            for sample, sample_path in sample_path_dict.items():
                print('Adding sample {}'.format(sample))
                self.__add_image(job, sample, sample_path)
            self.__commit(job)
            return wait_for_job(self.host, self.token, job)
        finally:
            os.remove(target_path)
        #
        # sample_names = list(pandas.read_csv(data_folder + 'target.csv').iloc[:, 0])
        # check_sample_names(sample_names)
        # for sample in sample_names:
        #     print('Adding sample {}'.format(sample))
        #     sample_path = find_sample_in_folder(data_folder, sample)
        #     self.__add_image(job, sample, sample_path)
        # self.__commit(job)
        # return wait_for_job(self.host, self.token, job)


    def __add_image(self, job: Job, sample: str, path: str):
        fname = path.split('/')[-1]
        mp_encoder = MultipartEncoder(
            fields={
                'taskId': str(job.id),
                'sampleId': sample,
                'file': (fname, open(path, 'rb'), 'text/plain')
            }
        )
        resp = requests.post(
            self.host + ADD_IMAGE_ENDPOINT,
            data=mp_encoder,
            headers={
                'Content-Type': mp_encoder.content_type,
                'Authorization': "Bearer" + " " + self.token
            },
            verify=True
        )
        if not resp.ok:
            print(resp.content)
            raise

    def __commit(self, job: Job):
        resp = requests.post(
            self.host + COMMIT_IMAGE_ENDPOINT,
            data=json.dumps({'id': job.id}),
            headers= json_headers(self.token),
            verify=True
        )
        if not resp.ok:
            print(resp.content)
            raise

    def __init_job(self, project: int, name: str, target_path: str, has_feature_names=True, description: str = None):
        data_file = target_path
        mp_encoder = MultipartEncoder(
            fields={
                'attributes': json_encode_image_init(project, name, description, has_feature_names),
                'file': ('target.csv', open(data_file, 'rb'), 'text/plain'),
            }
        )
        resp = requests.post(
            self.host + NEW_IMAGE_ENDPOINT,
            data=mp_encoder,
            headers={
                'Content-Type': mp_encoder.content_type,
                'Authorization': "Bearer" + " " + self.token
            },
            verify=True
        )
        if not resp.ok:
            print(resp.content)
            raise
        return Job(json.loads(resp.content)['id'])
