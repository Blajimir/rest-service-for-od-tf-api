import os
import json
import six.moves.urllib as urllib

PATH_TO_SAVE = '/pydir/models/research/object_detection/freeze-models'
if not os.path.exists(PATH_TO_SAVE):
    os.makedirs(PATH_TO_SAVE)

json_data = json.loads(open('model-list.json').read())
opener = urllib.request.URLopener()
len_list = len(json_data['list'])
print('start load models')
for ind, item in enumerate(json_data['list']):
    print('loading model {} number {} of {}'.format(item['name'], ind+1, len_list))
    opener.retrieve(item['link'], os.path.join(PATH_TO_SAVE, item['link'].split('/')[-1]))
print('all  models loaded')
