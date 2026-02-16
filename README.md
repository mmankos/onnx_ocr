# C++ OCR Pipeline utilizing ONNX

C++ OCR pipeline utilizing ONNX Runtime for optimized inference. Designed to
replace the official high-level language implementations with improved textbox
merging.

https://github.com/user-attachments/assets/0872c333-4dc0-4eff-9e78-f71caa83e123

## TODO
- [X] Character Detection
- [X] Directional Classifier
- [X] Character Recognition
- [ ] Dockerize

## Setup

### Pull PaddlePaddle docker image
Check out the
[paddlepaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/docker/fromdocker_en.html)
website to get the current version. At the time of writing this it was the
**paddlepaddle/paddle:3.2.0**.

If outside of china do not use the first recommended images containing
*cnc.bj.baidubce.com*, as they will download slower than the official Docker Hub
version.
```
docker pull paddlepaddle/paddle:3.2.0
```

### Build and enter Docker container
As we won't be needing the container after exporting the ONNX models we set the
`--rm` flag to automatically remove the container on exit.

If you plan on using the container for other tasks you can replace the `--rm`
with `--name YOUR_CONTAINER_NAME`.

`-v $(pwd)/onnx_models:/onnx_models` just makes sure that the models we convert
to ONNX will be persisted in the `current_path/onnx_models` on your machine.

**If you decide to pull a different image, not the *paddlepaddle/paddle:3.2.0*, replace
it in this command too.**
```
docker run --rm -it \
  -v "$PWD"/onnx_models:/workspace/onnx_models \
  paddlepaddle/paddle:3.2.0 \
  /bin/bash
```

### Obtain the ONNX models
Search through the model list sections for relevant [model download
links](https://www.paddleocr.ai/main/en/version3.x/module_usage/module_overview.html).

I will be using the mobile inference models as the score to file size and
execution time ratio is far better than the regular models.

#### Download the PaddleOCR models
```
mkdir -p /workspace/onnx_models && cd /workspace/onnx_models

pip install paddleocr onnxruntime
paddlex --install paddle2onnx

alias wxt='f(){
    mkdir -p "$1"
    wget "$2" -O "$1".tar
    mapfile -t top <<<"$(tar tf "$1".tar | cut -d/ -f1 | uniq)"

    if [[ ${#top[@]} -eq 1 ]]; then
        if tar tvf "$1".tar | grep "^d" | grep -m1 "${top[0]}"; then
            tar xf "$1".tar --strip-components=1 -C "$1"
        else
            tar xf "$1".tar -C "$1"
        fi
    else
        tar xf "$1".tar -C "$1"
    fi

    paddlex --paddle2onnx --paddle_model_dir "$1" --onnx_model_dir .
    mv inference.onnx "$1".onnx

    rm -rf "$1" "$1".tar *.yml
}; f'

# these links can be found on the https://www.paddleocr.ai/main/en/version3.x/module_usage/module_overview.html
wxt det https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar
#wxt rec https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0//PP-OCRv5_mobile_rec_infer.tar
wxt rec https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/latin_PP-OCRv5_mobile_rec_infer.tar
wxt cls https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar
```

#### Convert the saved models to ONNX
```
pip install paddleocr
paddlex --install paddle2onnx
cd onnx_models
```
```
paddlex --paddle2onnx --paddle_model_dir cls --onnx_model_dir .
mv inference.onnx det.onnx
```
```
paddlex \
     --paddle2onnx \
     --paddle_model_dir PP-OCRv5_mobile_rec_infer \
     --onnx_model_dir .
mv inference.onnx rec.onnx
```
```
paddlex \
     --paddle2onnx \
     --paddle_model_dir PP-LCNet_x1_0_textline_ori_infer \
     --onnx_model_dir .
mv inference.onnx cls.onnx
```

```
exit
ls onnx_models
```

should return
```
det.onnx  inference.yml  rec.onnx
```

#### Download the dictionary for Character Recognition
```
wget -O dict.txt https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/main/ppocr/utils/dict/ppocrv5_en_dict.txt
```

## Usage
1. Update the `config.yaml` to match your settings and filepaths
2.
```
$ make
```
