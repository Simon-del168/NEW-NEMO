# NEW-NEMO
update for NEMO:https://github.com/kaist-ina/nemo
```
@inproceedings{yeo2020nemo,
  title={NEMO: enabling neural-enhanced video streaming on commodity mobile devices},
  author={Yeo, Hyunho and Chong, Chan Ju and Jung, Youngmok and Ye, Juncheol and Han, Dongsu},
  booktitle={Proceedings of the 26th Annual International Conference on Mobile Computing and Networking},
  pages={1--14},
  year={2020}
}
```
## Prerequisites
We cannot provide this due to the Qualcom license policy.Please contract me!
## Guide
Because the time flies,many errors occured.So do some update.
### 1.Setup
* clone the docker contianer build repository
```
https://github.com/Simon-del168/NEW-NEMO/tree/main/Setup
```
* Build the image
```
cd ${HOME}/Setup
./build.sh
```
* Run & Attach to the docker
```
cd ${HOME}/Setup
./run.sh
```
Download/Setup the Qualcomm SNPE SDK as follow:
```
./nemo
├── video                  # Python: Video downloader/encoder
├── dnn                    # Python: DNN trainer/converter
├── cache_profile          # Python: Anchor point selector
├── player                 # Java, C/C++: Android video player built upon Exoplayer and the SR-integrated codec
./third_party
├── libvpx                 # C/C++: SR-integrated codec
```
### 2. Prepare videos

* Download a Youtube video
```
$NEMO_CODE_ROOT/nemo/tool/script/download_video.sh -c product_review
```
* Encode the video 
```
$NEMO_CODE_ROOT/nemo/tool/script/encode_video.sh -c product_review
```
### 3. Prepare DNNs

* Train a DNN
```
$NEMO_CODE_ROOT/nemo/dnn/script/train_video.sh -g 0 -c product_review -q high -i 240 -o 1080
```

* (Optional) Convert the TF model to the Qualcomm SNPE dlc
```
$NEMO_CODE_ROOT/nemo/dnn/script/convert_tf_to_snpe.sh -g 0 -c product_review -q high -i 240 -o 1080
```

* (Optional) Test the dlc on Qualcomm devices
```
$NEMO_CODE_ROOT/nemo/dnn/script/test_snpe.sh -g 0 -c product_review -q high -r 240 -s 4 -d [device id]
```
[Details are described in this file.](nemo/dnn/README.md)

### 4. Generate a cache profile 

* Setup: Build the SR-integrated codec (x86_64)
```
$NEMO_CODE_ROOT/nemo/cache_profile/script/setup.sh
```

* Generate the cache profile using the codec
```
$NEMO_CODE_ROOT/nemo/cache_profile/script/select_anchor_points.sh -g 0 -c product_review -q high -i 240 -o 1080 -a nemo
```

* (Optional) Analyze frame dependencies & frame types
```
$NEMO_CODE_ROOT/nemo/cache_profile/script/analyze_video.sh -g 0 -c product_review -q high -i 240 -o 1080 -a nemo
```
[Details are described in this file.](nemo/cache_profile/README.md)

### 5. Compare NEMO vs. baselines
* Setup: Build the SR-integrated codec (arm64-v8)
```
$NEMO_CODE_ROOT/nemo/test/script/setup_local.sh 
```
* Setup: Copy data to mobile devices 
```
$NEMO_CODE_ROOT/nemo/test/script/setup_device.sh -g 0 -c product_review -q high -r 240 -a nemo_0.5 -d [device id]
```
* Measure the latency
```
$NEMO_CODE_ROOT/nemo/test/script/measure_latency.sh -g 0 -c product_review -q high -r 240 -a nemo_0.5 -d [device id]
```
* Measure the quality
```
$NEMO_CODE_ROOT/nemo/test/script/measure_quality.sh -g 0 -c product_review -q high -i 240 -o 1080 -a nemo_0.5 
```
[Details are described in this file.](nemo/test/README.md)

### 6. Play NEMO in Android smartphones 
* Setup: Copy data to mobile devices
```
$NEMO_CODE_ROOT/nemo/player/script/setup_device.sh -g 0 -c product_review -q high -r 240 -a nemo_0.5 -d [device id] -a nemo_0.5_16
```

