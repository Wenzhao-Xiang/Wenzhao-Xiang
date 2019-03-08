const videoElement = document.getElementById('video');
const imageElement = document.getElementById('image');
const inputElement = document.getElementById('input');
const canvasElement = document.getElementById('canvas');
const canvasShowElement = document.getElementById('canvasshow');
const progressBar = document.getElementById('progressBar');

let currentBackend = getSearchParamsBackend();
let currentModel = getSearchParamsModel();
let currentPrefer = getSearchParamsPrefer();
let streaming = false;
let stats = new Stats();
let track;

const showAlert = (error) => {
  console.error(error);
  let div = document.createElement('div');
  // div.setAttribute('id', 'backendAlert');
  div.setAttribute('class', 'backendAlert alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = `<strong>${error}</strong>`;
  div.innerHTML += `<button type='button' class='close' data-dismiss='alert' aria-label='Close'><span aria-hidden='true'>&times;</span></button>`;
  let container = document.getElementById('container');
  container.insertBefore(div, container.firstElementChild);
}

const updateProgress = (ev) => {
  if (ev.lengthComputable) {
    let totalSize = ev.total / (1000 * 1000);
    let loadedSize = ev.loaded / (1000 * 1000);
    let percentComplete = ev.loaded / ev.total * 100;    
    percentComplete = percentComplete.toFixed(0);
    progressBar.style = `width: ${percentComplete}%`;
    updateLoading(loadedSize.toFixed(1), totalSize.toFixed(1), percentComplete);
  }
}

let utils = new Utils(canvasElement, canvasShowElement);
utils.updateProgress = updateProgress;    //register updateProgress function if progressBar element exist

const updateResult = (result) => {
  try {
    console.log(`Inference time: ${result.time} ms`);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `inference time: <span class='ir'>${result.time} ms</span>`;
  } catch(e) {
    console.log(e);
  }
}

const logConfig = () => {
  console.log(`Model: ${currentModel}, Backend: ${currentBackend}, Prefer: ${currentPrefer}`);
}

const errorHandler = (e) => {
  showAlert(e);
  showError(null, null);
}

const startPredictCamera = async () => {
  if (streaming) {
    try {
      stats.begin();
      let ret = await utils.predict(videoElement);
      updateResult(ret);
      stats.end();
      setTimeout(startPredictCamera, 0);
    } catch (e) {
      errorHandler(e);
    }
  }
}

const utilsPredict = async (imageElement, backend, prefer) => {
  streaming = false;
  // Stop webcam opened by navigator.getUserMedia if user visits 'LIVE CAMERA' tab before
  if(track) {
    track.stop();
  }
  try {
    await showProgress('Image inferencing ...');
    let ret = await utils.predict(imageElement);
    showResults();
    updateResult(ret);
  }
  catch (e) {
    errorHandler(e);
  }
}

videoElement.addEventListener('loadeddata', () => {
  startPredictCamera();
  showResults();
}, false);

const utilsPredictCamera = async (backend, prefer) => {
  streaming = true;
  try {
    let stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: 'environment' } });
    video.srcObject = stream;
    track = stream.getTracks()[0];
    showProgress('Camera inferencing ...');
  }
  catch (e) {
    errorHandler(e);
  }
}

const predictPath = (camera) => {
  (!camera) ? utilsPredict(imageElement, currentBackend, currentPrefer) : utilsPredictCamera(currentBackend, currentPrefer);
}

const utilsInit = async (backend, prefer) => {
  // return immediately if model, backend, prefer are all unchanged
  let init = await utils.init(backend, prefer);    
  if (init == 'NOT_LOADED') {
    return;
  }
}

const updateScenario = async (camera = false) => {
  streaming = false;
  logConfig();
  predictPath(camera);
}

 const updateBackend = async (camera = false) => {
  streaming = false;
  logConfig();
  try {
    await utilsInit(currentBackend, currentPrefer);
    predictPath(camera);
  }
  catch (e) {
    errorHandler(e);
  }
}

inputElement.addEventListener('change', (e) => {
  let files = e.target.files;
  if (files.length > 0) {
    imageElement.src = URL.createObjectURL(files[0]);
  }
}, false);

imageElement.addEventListener('load', () => {
  utilsPredict(imageElement, currentBackend, currentPrefer);
}, false);

const main = async (camera = false) => {
  streaming = false;
  try { utils.deleteAll(); } catch (e) {}
  logConfig();
  await showProgress('Loading model ...');
  try {
    let model = objectDetectionModels.filter(f => f.modelFormatName == currentModel);
    await utils.loadModel(model[0]);
    await utilsInit(currentBackend, currentPrefer);
  } catch (e) {
    errorHandler(e);
  }
  predictPath(camera);
}

