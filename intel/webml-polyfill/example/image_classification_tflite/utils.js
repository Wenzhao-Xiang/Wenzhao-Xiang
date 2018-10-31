const mobilenet_v1 = {
  MODEL_NAME : 'Mobilenet_V1',
  INPUT_SIZE : 224,
  OUTPUT_SIZE : 1001,
  MODEL_FILE : './model/mobilenet_v1_1.0_224.tflite',
  LABELS_FILE : './model/labels.txt'
};
const mobilenet_v2 = {
  MODEL_NAME : 'Mobilenet_V2',
  INPUT_SIZE : 224,
  OUTPUT_SIZE : 1001,
  MODEL_FILE : './model/mobilenet_v2_1.0_224.tflite',
  LABELS_FILE : './model/labels.txt'
};
const inception_v3 = {
  MODEL_NAME : 'Inception_V3',
  INPUT_SIZE : 299,
  OUTPUT_SIZE : 1001,
  MODEL_FILE : './model/inception_v3.tflite',
  LABELS_FILE : './model/labels.txt'
};
const squeezenet = {
  MODEL_NAME : 'Squeezenet',
  INPUT_SIZE : 224,
  OUTPUT_SIZE : 1001,
  MODEL_FILE : './model/squeezenet.tflite',
  LABELS_FILE : './model/labels.txt'
}

class Utils {
  constructor() {
    this.tfModel;
    this.labels;
    this.model;
    this.inputTensor;
    this.outputTensor;

    this.container = document.getElementById('container');
    this.progressBar = document.getElementById('progressBar');
    this.progressContainer = document.getElementById('progressContainer');
    this.canvasElement = document.getElementById('canvas');
    this.canvasContext = this.canvasElement.getContext('2d');

    this.initialized = false;
  }

  async init(backend) {
    this.initialized = false;
    let result;
    if (!this.tfModel) {
      result = await this.loadModelAndLabels(MODEL_FILE, LABELS_FILE);
      progressContainer.style.display = "none";
      progressBar.style = `width: 0%`;
      progressBar.innerHTML = `0%`;
      this.labels = result.text.split('\n');
      console.log(`labels: ${this.labels}`);
      let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
      this.tfModel = tflite.Model.getRootAsModel(flatBuffer);
      printTfLiteModel(this.tfModel);
    }
    this.model = new ImageClassificationModel(this.tfModel, backend);
    result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;
  }

  async predict(imageSource) {
    if (!this.initialized) return;
    this.canvasContext.drawImage(imageSource, 0, 0,
                                 this.canvasElement.width,
                                 this.canvasElement.height);
    this.prepareInputTensor(this.inputTensor, this.canvasElement);
    let start = performance.now();
    let result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    let classes = this.getTopClasses(this.outputTensor, this.labels, 3);
    console.log(`Inference time: ${elapsed.toFixed(2)} ms`);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `inference time: <em style="color:green;font-weight:bloder;">${elapsed.toFixed(2)} </em>ms`;
    console.log(`Classes: `);
    classes.forEach((c, i) => {
      console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
      let labelElement = document.getElementById(`label${i}`);
      let probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = `${c.label}`;
      probElement.innerHTML = `${c.prob}%`;
    });
  }

  async loadModelAndLabels(modelUrl, labelsUrl) {
    let arrayBuffer = await this.loadUrl(modelUrl, true, true);
    let bytes = new Uint8Array(arrayBuffer);
    let text = await this.loadUrl(labelsUrl);
    return {bytes: bytes, text: text};
  }

  async loadUrl(url, binary, progress) {
    return new Promise((resolve, reject) => {
      let request = new XMLHttpRequest();
      request.open('GET', url, true);
      if (binary) {
        request.responseType = 'arraybuffer';
      }
      request.onload = function(ev) {
        if (request.readyState === 4) {
          if (request.status === 200) {
              resolve(request.response);
          } else {
              reject(new Error('Failed to load ' + modelUrl + ' status: ' + request.status));
          }
        }
      };
      if (progress) {
        let self = this;
        request.onprogress = function(ev) {
          if (ev.lengthComputable) {
            let percentComplete = ev.loaded / ev.total * 100;
            percentComplete = percentComplete.toFixed(0);
            self.progressBar.style = `width: ${percentComplete}%`;
            self.progressBar.innerHTML = `${percentComplete}%`;
          }
        }
      }
      request.send();
    });
  }

  prepareInputTensor(tensor, canvas) {
    const width = INPUT_SIZE;
    const height = INPUT_SIZE;
    const channels = 3;
    const imageChannels = 4; // RGBA
    const mean = 127.5;
    const std = 127.5;
    if (canvas.width !== width || canvas.height !== height) {
      throw new Error(`canvas.width(${canvas.width}) or canvas.height(${canvas.height}) is not ${INPUT_SIZE}`);
    }
    let context = canvas.getContext('2d');
    let pixels = context.getImageData(0, 0, width, height).data;
    // NHWC layout
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y*width*imageChannels + x*imageChannels + c];
          tensor[y*width*channels + x*channels + c] = (value - mean)/std;
        }
      }
    }
  }

  getTopClasses(tensor, labels, k = 5) {
    let probs = Array.from(tensor);
    let indexes = probs.map((prob, index) => [prob, index]);
    let sorted = indexes.sort((a, b) => {
      if (a[0] === b[0]) {return 0;}
      return a[0] < b[0] ? -1 : 1;
    });
    sorted.reverse();
    let classes = [];
    for (let i = 0; i < k; ++i) {
      let prob = sorted[i][0];
      let index = sorted[i][1];
      let c = {
        label: labels[index],
        prob: (prob * 100).toFixed(2)
      }
      classes.push(c);
    }
    return classes;
  }

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }
}
