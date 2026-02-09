## Pull PaddlePaddle docker image
Check the
<https://www.paddlepaddle.org.cn/documentation/docs/en/install/docker/fromdocker_en.html>
to get the current version. At the time of writing this it was the
**paddlepaddle/paddle:3.2.0**.

If outside of china do not use the first recommended images *containing
cnc.bj.baidubce.com*, as they will download slower than the official Docker Hub
version.
```
docker pull paddlepaddle/paddle:3.2.0
```

## Build and enter Docker container
As we won't be needing the container after exporting the ONNX models we set the
`--rm` flag to automatically remove the container on exit.

If you plan on using the container for other tasks you can replace the `--rm`
with `--name YOUR_CONTAINER_NAME`.

`-v $(pwd)/onnx_models:/onnx_models` just makes sure that the models we convert
to ONNX will be persisted in the `current_path/onnx_models` on your machine.

**If you decide to pull a different image, not the paddlepaddle/paddle:3.2.0, replace
it in this command too.**
```
docker run --rm -it \
  -v "$PWD"/onnx_models:/workspace/onnx_models \
  paddlepaddle/paddle:3.2.0 \
  /bin/bash
```

## Obtain the ONNX models
https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/deployment/cpp/OCR.en.md#12-compile-paddle-inference

### Download the PaddleOCR models
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

wxt det https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar
#wxt rec https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0//PP-OCRv5_mobile_rec_infer.tar
wxt rec https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/latin_PP-OCRv5_mobile_rec_infer.tar
wxt cls https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar
```

### Convert the saved models to ONNX
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

https://github.com/microsoft/onnxruntime/releases

??? git submodule add ???
```
wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
tar -xzf onnxruntime-linux-x64-1.23.2.tgz
```


```
wget -O dict.txt https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/main/ppocr/utils/dict/ppocrv5_en_dict.txt

```

https://toon-beerten.medium.com/ocr-comparison-tesseract-versus-easyocr-vs-paddleocr-vs-mmocr-a362d9c79e66

https://jinscott.medium.com/onnx-runtime-on-c-67f69de9b95c

https://www.kaggle.com/code/karensnchez/fine-tuning-paddleocr-det-model
