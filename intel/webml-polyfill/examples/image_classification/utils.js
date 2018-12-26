class Utils {
  constructor(canvas) {
    this.rawModel;
    this.labels;
    this.model;
    this.inputTensor;
    this.outputTensor;
    this.modelFile;
    this.labelsFile;
    this.inputSize;
    this.outputSize;
    this.preOptions;
    this.postOptions;
    this.canvasElement = canvas;
    this.canvasContext = this.canvasElement.getContext('2d');
    this.updateProgress;
    this.initialized = false;
  }

  async init(backend, prefer) {
    this.initialized = false;
    let result;
    let kwargs = {
      rawModel: null,
      backend: backend,
      prefer: prefer,
      softmax: this.postOptions.softmax || false,
    };
    if (this.modelFile.split('.').pop() === 'tflite') {
      if (!this.rawModel) {
        result = await this.loadModelAndLabels(this.modelFile, this.labelsFile);
        this.labels = result.text.split('\n');
        console.log(`labels: ${this.labels}`);
        let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
        this.rawModel = tflite.Model.getRootAsModel(flatBuffer);
        printTfLiteModel(this.rawModel);
      }
      kwargs.rawModel = this.rawModel;
      this.model = new TFliteModelImporter(kwargs);
    } else if (this.modelFile.split('.').pop() === 'onnx') {
      if (!this.rawModel) {
        result = await this.loadModelAndLabels(this.modelFile, this.labelsFile);
        this.labels = result.text.split('\n');
        console.log(`labels: ${this.labels}`);
        let err = onnx.ModelProto.verify(result.bytes);
        if (err) {
          throw new Error(`Invalid model ${err}`);
        }
        this.rawModel = onnx.ModelProto.decode(result.bytes);
        printOnnxModel(this.rawModel);
      }
      kwargs.rawModel = this.rawModel;
      this.model = new OnnxModelImporter(kwargs);
    }
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
    return {
      time: elapsed.toFixed(2),
      classes: this.getTopClasses(this.outputTensor, this.labels, 3)
    };
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
      if (progress && typeof this.updateProgress !== 'undefined') {
        request.onprogress = this.updateProgress;
      }
      request.send();
    });
  }

  prepareInputTensor(tensor, canvas) {
    const width = this.inputSize[1];
    const height = this.inputSize[0];
    const channels = this.inputSize[2];
    const imageChannels = 4; // RGBA
    const mean = this.preOptions.mean || [0, 0, 0, 0];
    const std  = this.preOptions.std  || [1, 1, 1, 1];
    const norm = this.preOptions.norm || false;
    const channelScheme = this.preOptions.channelScheme || 'RGB';
    if (canvas.width !== width || canvas.height !== height) {
      throw new Error(`canvas.width(${canvas.width}) is not ${width} or canvas.height(${canvas.height}) is not ${height}`);
    }
    let context = canvas.getContext('2d');
    let pixels = context.getImageData(0, 0, width, height).data;
    if (norm) {
      pixels = new Float32Array(pixels).map(p => p / 255);
    }
    
    if (channelScheme === 'RGB') {
      // NHWC layout
      for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
          for (let c = 0; c < channels; ++c) {
            let value = pixels[y*width*imageChannels + x*imageChannels + c];
            tensor[y*width*channels + x*channels + c] = (value - mean[c]) / std[c];
          }
        }
      }
    } else if (channelScheme === 'BGR') {
      // NHWC layout
      for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
          for (let c = 0; c < channels; ++c) {
            let value = pixels[y*width*imageChannels + x*imageChannels + (channels-c-1)];
            tensor[y*width*channels + x*channels + c] = (value - mean[c]) / std[c];
          }
        }
      }
    } else {
      throw new Error(`Unknown color channel scheme ${channelScheme}`);
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

  changeModelParam(newModel) {
    this.inputSize = newModel.inputSize;
    this.outputSize = newModel.outputSize;
    this.modelFile = newModel.modelFile;
    this.labelsFile = newModel.labelsFile;
    this.preOptions = newModel.preOptions || {};
    this.postOptions = newModel.postOptions || {};
    this.inputTensor = new Float32Array(this.inputSize.reduce((a, b) => a * b));
    this.outputTensor = new Float32Array(this.outputSize);
    this.rawModel = null;

    this.canvasElement.width = newModel.inputSize[1];
    this.canvasElement.height = newModel.inputSize[0];
  }
}
