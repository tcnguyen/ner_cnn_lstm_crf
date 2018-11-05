- The Elmo model from tensorflow_hub is put to: trongcanh/datasets/tfhubcache/1
- Dataset: trongcanh/datasets/ner/2

floyd run \
--data trongcanh/datasets/ner/2:/dataCoNLL2003 \
--data trongcanh/datasets/tfhubcache/1:/dataTFHubCache \
--gpu --env tensorflow-1.11 "python floydhub.py"