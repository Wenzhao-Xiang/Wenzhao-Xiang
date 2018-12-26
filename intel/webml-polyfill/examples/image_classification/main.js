const mobilenet_v1_tflite = {
  modelName: 'Mobilenet V1(TFlite)',
  inputSize: [224, 224, 3],
  outputSize: 1001,
  modelFile: './model/mobilenet_v1_1.0_224.tflite',
  labelsFile: './model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  }
};
const mobilenet_v2_tflite = {
  modelName: 'Mobilenet V2(TFlite)',
  inputSize: [224, 224, 3],
  outputSize: 1001,
  modelFile: './model/mobilenet_v2_1.0_224.tflite',
  labelsFile: './model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  }
};
const inception_v3_tflite = {
  modelName: 'Inception V3(TFlite)',
  inputSize: [299, 299, 3],
  outputSize: 1001,
  modelFile: './model/inception_v3.tflite',
  labelsFile: './model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
};
const inception_v4_tflite = {
  modelName: 'Inception V4(TFlite)',
  inputSize: [299, 299, 3],
  outputSize: 1001,
  modelFile: './model/inception_v4.tflite',
  labelsFile: './model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  }
};
const squeezenet_tflite = {
  modelName: 'Squeezenet(TFlite)',
  inputSize: [224, 224, 3],
  outputSize: 1001,
  modelFile: './model/squeezenet.tflite',
  labelsFile: './model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  }
};
const inception_resnet_v2_tflite = {
  modelName: 'Incep. Res. V2(TFlite)',
  inputSize: [299, 299, 3],
  outputSize: 1001,
  modelFile: './model/inception_resnet_v2.tflite',
  labelsFile: './model/labels1001.txt',
  preOptions: {
    mean: [127.5, 127.5, 127.5],
    std: [127.5, 127.5, 127.5],
  },
  postOptions: {
    softmax: true,
  }
};
const squeezenet_onnx = {
  modelName: 'SqueezeNet(Onnx)',
  modelFile: './model/squeezenet1.1.onnx',
  labelsFile: './model/labels1000.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  preOptions: {
    // https://github.com/onnx/models/tree/master/models/image_classification/squeezenet#preprocessing
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    norm: true
  },
  postOptions: {
    softmax: true,
  }
};
const mobilenet_v2_onnx = {
  modelName: 'Mobilenet v2(Onnx)',
  modelFile: './model/mobilenetv2-1.0.onnx',
  labelsFile: './model/labels1000.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  preOptions: {
    // https://github.com/onnx/models/tree/master/models/image_classification/mobilenet#preprocessing
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    norm: true
  },
  postOptions: {
    softmax: true,
  }
};
const resnet_v1_onnx = {
  modelName: 'ResNet50 v1(Onnx)',
  modelFile: './model/resnet50v1.onnx',
  labelsFile: './model/labels1000.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  preOptions: {
    // https://github.com/onnx/models/tree/master/models/image_classification/resnet#preprocessing
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    norm: true
  },
  postOptions: {
    softmax: true,
  }
};
const resnet_v2_onnx = {
  modelName: 'ResNet50 v2(Onnx)',
  modelFile: './model/resnet50v2.onnx',
  labelsFile: './model/labels1000.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  preOptions: {
    // https://github.com/onnx/models/tree/master/models/image_classification/resnet#preprocessing
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    norm: true
  },
  postOptions: {
    softmax: true,
  }
};
const inceptionv2_onnx = {
  modelName: 'Inception v2(Onnx)',
  modelFile: './model/inceptionv2.onnx',
  labelsFile: './model/ilsvrc2012labels.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
};
const densenet_onnx = {
  modelName: 'DenseNet(Onnx)',
  modelFile: './model/densenet121.onnx',
  labelsFile: './model/labels1000.txt',
  inputSize: [224, 224, 3],
  outputSize: 1000,
  preOptions: {
    // mean and std should also be in BGR order
    mean: [0.406, 0.456, 0.485],
    std: [0.225, 0.224, 0.229],
    norm: true,
    channelScheme: 'BGR',
  },
  postOptions: {
    softmax: true,
  }
};
const preferMap = {
  'MPS': 'sustained',
  'BNNS': 'fast',
  'sustained': 'MPS',
  'fast': 'BNNS',
};

function main(camera) {
  const availableModels = [
    mobilenet_v1_tflite,
    mobilenet_v2_tflite,
    inception_v3_tflite,
    inception_v4_tflite,
    squeezenet_tflite,
    inception_resnet_v2_tflite,
    squeezenet_onnx,
    mobilenet_v2_onnx,
    resnet_v1_onnx,
    resnet_v2_onnx,
    inceptionv2_onnx,
    densenet_onnx,
  ];

  const videoElement = document.getElementById('video');
  const imageElement = document.getElementById('image');
  const inputElement = document.getElementById('input');
  const buttonEelement = document.getElementById('button');
  const backend = document.getElementById('backend');
  const selectModel = document.getElementById('selectModel');
  const wasm = document.getElementById('wasm');
  const webgl = document.getElementById('webgl');
  const webml = document.getElementById('webml');
  const canvasElement = document.getElementById('canvas');
  const progressContainer = document.getElementById('progressContainer');
  const progressBar = document.getElementById('progressBar');
  const selectPrefer = document.getElementById('selectPrefer');

  let currentBackend = '';
  let currentModel = '';
  let currentPrefer = '';
  let streaming = false;

  let utils = new Utils(canvasElement);
  utils.updateProgress = updateProgress;    //register updateProgress function if progressBar element exist

  function checkPreferParam() {
    if (getOS() === 'Mac OS') {
      let preferValue = getPreferParam();
      if (preferValue === 'invalid') {
        console.log("Invalid prefer, prefer should be 'fast' or 'sustained', try to use WASM.");
        showPreferAlert();
      }
    }
  }

  checkPreferParam();

  function showAlert(backend) {
    let div = document.createElement('div');
    div.setAttribute('id', 'backendAlert');
    div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
    div.setAttribute('role', 'alert');
    div.innerHTML = `<strong>Failed to setup ${backend} backend.</strong>`;
    div.innerHTML += `<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>`;
    let container = document.getElementById('container');
    container.insertBefore(div, container.firstElementChild);
  }

  function showPreferAlert() {
    let div = document.createElement('div');
    div.setAttribute('id', 'preferAlert');
    div.setAttribute('class', 'alert alert-danger alert-dismissible fade show');
    div.setAttribute('role', 'alert');
    div.innerHTML = `<strong>Invalid prefer, prefer should be 'fast' or 'sustained'.</strong>`;
    div.innerHTML += `<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>`;
    let container = document.getElementById('container');
    container.insertBefore(div, container.firstElementChild);
  }

  function removeAlertElement() {
    let backendAlertElem =  document.getElementById('backendAlert');
    if (backendAlertElem !== null) {
      backendAlertElem.remove();
    }
    let preferAlertElem =  document.getElementById('preferAlert');
    if (preferAlertElem !== null) {
      preferAlertElem.remove();
    }
  }

  function updateBackend() {
    if (getUrlParams('api_info') === 'true') {
      backend.innerHTML = currentBackend === 'WebML' ? currentBackend + '/' + getNativeAPI() : currentBackend;
    } else {
      backend.innerHTML = currentBackend;
    }
  }

  function changeBackend(newBackend, force) {
    if (!force && currentBackend === newBackend) {
      return;
    }
    streaming = false;
    if (newBackend !== "WebML") {
      selectPrefer.style.display = 'none';
    } else {
      selectPrefer.style.display = 'inline';
    }
    utils.deleteAll();
    backend.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(newBackend, currentPrefer).then(() => {
        currentBackend = newBackend;
        updatePrefer();
        updateModel();
        updateBackend();
        if (!camera) {
          utils.predict(imageElement).then(ret => updateResult(ret));
        } else {
          streaming = true;
          startPredict();
        }
      }).catch((e) => {
        console.warn(`Failed to change backend ${newBackend}, switch back to ${currentBackend}`);
        console.log(e);
        showAlert(newBackend);
        changeBackend(currentBackend, true);
        updatePrefer();
        updateModel();
        updateBackend();
      });
    }, 10);
  }

  function changeModel(newModel) {
    if (currentModel === newModel.modelName) {
      return;
    }
    streaming = false;
    utils.deleteAll();
    utils.changeModelParam(newModel);
    progressContainer.style.display = "inline";
    currentPrefer = "sustained";
    selectModel.innerHTML = 'Setting...';
    currentModel = newModel.modelName;
    setTimeout(() => {
      utils.init(currentBackend, currentPrefer).then(() => {
        updatePrefer();
        updateModel();
        updateBackend();
        if (!camera) {
          utils.predict(imageElement).then(ret => updateResult(ret));
        } else {
          streaming = true;
          startPredict();
        }
      });
    }, 10);
  }

  function updateModel() {
    selectModel.innerHTML = currentModel;
  }

  function changePrefer(newPrefer, force) {
    if (currentPrefer === newPrefer && !force) {
      return;
    }
    streaming = false;
    utils.deleteAll();
    selectPrefer.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(currentBackend, newPrefer).then(() => {
        currentPrefer = newPrefer;
        updatePrefer();
        updateModel();
        updateBackend();
        if (!camera) {
          utils.predict(imageElement).then(ret => updateResult(ret));
        } else {
          streaming = true;
          startPredict();
        }
      }).catch((e) => {
        console.warn(`Failed to change backend ${preferMap[newPrefer]}, switch back to ${preferMap[currentPrefer]}`);
        console.error(e);
        showAlert(preferMap[newPrefer]);
        changePrefer(currentPrefer, true);
        updatePrefer();
        updateModel();
        updateBackend();
      });
    }, 10);
  }

  function updatePrefer() {
    selectPrefer.innerHTML = preferMap[currentPrefer];
  }

  function fileExists(url) {
    var exists;
    $.ajax({
      url:url,
      async:false,
      type:'HEAD',
      error:function() { exists = 0; },
      success:function() { exists = 1; }
    });
    if (exists === 1) {
      return true;
    } else {
      return false;
    }
  }

  function updateProgress(ev) {
    if (ev.lengthComputable) {
      let percentComplete = ev.loaded / ev.total * 100;
      percentComplete = percentComplete.toFixed(0);
      progressBar.style = `width: ${percentComplete}%`;
      progressBar.innerHTML = `${percentComplete}%`;
      if (ev.loaded === ev.total) {
        progressContainer.style.display = "none";
        progressBar.style = `width: 0%`;
        progressBar.innerHTML = `0%`;
      }
    }
  }

  function updateResult(result) {
    console.log(`Inference time: ${result.time} ms`);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `inference time: <em style="color:green;font-weight:bloder;">${result.time} </em>ms`;
    console.log(`Classes: `);
    result.classes.forEach((c, i) => {
      console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
      let labelElement = document.getElementById(`label${i}`);
      let probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = `${c.label}`;
      probElement.innerHTML = `${c.prob}%`;
    });
  }
 
  if (nnNative) {
    webml.setAttribute('class', 'dropdown-item');
    webml.onclick = function (e) {
      removeAlertElement();
      checkPreferParam();
      changeBackend('WebML');
    };
  }

  if (nnPolyfill.supportWebGL) {
    webgl.setAttribute('class', 'dropdown-item');
    webgl.onclick = function(e) {
      removeAlertElement();
      changeBackend('WebGL');
    };
  }

  if (nnPolyfill.supportWasm) {
    wasm.setAttribute('class', 'dropdown-item');
    wasm.onclick = function(e) {
      removeAlertElement();
      changeBackend('WASM');
    };
  }

  if (currentBackend === '') {
    if (nnNative) {
      currentBackend = 'WebML';
    } else {
      currentBackend = 'WASM';
    }
  }

  // register models
  for (let model of availableModels) {
    if (!fileExists(model.modelFile)) {
      continue;
    }
    let dropdownBtn = $('<button class="dropdown-item"/>')
      .text(model.modelName)
      .click(_ => changeModel(model));
    $('.available-models').append(dropdownBtn);
    if (!currentModel) {
      utils.changeModelParam(model);
      currentModel = model.modelName;
    }
  }

  // register prefers
  if (getOS() === 'Mac OS' && currentBackend === 'WebML') {
    $('.prefer').css("display","inline");
    let MPS = $('<button class="dropdown-item"/>')
      .text('MPS')
      .click(_ => changePrefer(preferMap['MPS']));
    $('.preference').append(MPS);
    let BNNS = $('<button class="dropdown-item"/>')
      .text('BNNS')
      .click(_ => changePrefer(preferMap['BNNS']));
    $('.preference').append(BNNS);
    if (!currentPrefer) {
      currentPrefer = "sustained";
    }
  }

  // image or camera
  if (!camera) {
    inputElement.addEventListener('change', (e) => {
      let files = e.target.files;
      if (files.length > 0) {
        imageElement.src = URL.createObjectURL(files[0]);
      }
    }, false);

    imageElement.onload = function() {
      utils.predict(imageElement).then(ret => updateResult(ret));
    }
    utils.init(currentBackend, currentPrefer).then(() => {
      updateBackend();
      updateModel();
      updatePrefer();
      utils.predict(imageElement).then(ret => updateResult(ret));
      buttonEelement.setAttribute('class', 'btn btn-primary');
      inputElement.removeAttribute('disabled');
    }).catch((e) => {
      console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
      console.error(e);
      showAlert(utils.model._backend);
      changeBackend('WASM');
    });
  } else {
    let stats = new Stats();
    stats.dom.style.cssText = 'position:fixed;top:60px;left:10px;cursor:pointer;opacity:0.9;z-index:10000';
    stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
    document.body.appendChild(stats.dom);

    navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: "environment"}}).then((stream) => {
      video.srcObject = stream;
      utils.init(currentBackend, currentPrefer).then(() => {
        updateBackend();
        updateModel();
        updatePrefer();
        streaming = true;
        startPredict();
      }).catch((e) => {
        console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
        console.error(e);
        showAlert(utils.model._backend);
        changeBackend('WASM');
      });
    }).catch((error) => {
      console.log('getUserMedia error: ' + error.name, error);
    });

    function startPredict() {
      if (streaming) {
        stats.begin();
        utils.predict(videoElement).then(ret => updateResult(ret)).then(() => {
          stats.end();
          setTimeout(startPredict, 0);
        });
      }
    }
  }
}
