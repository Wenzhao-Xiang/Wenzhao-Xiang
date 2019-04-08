let guiState = {
  algorithm: 'multi-pose',
  model: 1.01,
  useAtrousConv: true, 
  outputStride: 16,
  scaleFactor: 0.5,
  scoreThreshold: 0.15,
  multiPoseDetection: {
    maxDetections: 15,
    nmsRadius: 20.0,
  },
  showPose: true,
  showBoundingBox: false,
};

const gui = new dat.GUI();
gui.domElement.id = 'gui';
const model = gui.add(guiState, 'model', [0.50, 0.75, 1.00, 1.01]);
const outputStride = gui.add(guiState, 'outputStride', [8, 16, 32]);
const scaleFactor = gui.add(guiState, 'scaleFactor', [0.25, 0.5, 0.75, 1.00]).listen();
const scoreThreshold = gui.add(guiState, 'scoreThreshold', 0.0, 1.0);
const multiPoseDetection = gui.addFolder('Multi Pose Estimation');
multiPoseDetection.open();
const nmsRadius = multiPoseDetection.add(guiState.multiPoseDetection, 'nmsRadius', 0.0, 40.0);
const maxDetections = multiPoseDetection.add(guiState.multiPoseDetection, 'maxDetections')
  .min(1)
  .max(20)
  .step(1);
const showPose = gui.add(guiState, 'showPose');
const showBoundingBox = gui.add(guiState, 'showBoundingBox');
const useAtrousConv = gui.add(guiState, 'useAtrousConv');
gui.close();
let customContainer = document.getElementById('my-gui-container');
customContainer.appendChild(gui.domElement);
guiState.scoreThreshold = 0.15;

let currentTab = 'image';

const utils = new Utils();
const canvassingle = document.getElementById('canvas');
const ctxSingle = canvassingle.getContext('2d');
const canvasmulti = document.getElementById('canvasmulti');
const ctxMulti = canvasmulti.getContext('2d');
const scaleImage = document.getElementById('scaleimage');
const scaleCanvas = document.getElementById('scalevideo');
const video = document.getElementById('video');
const canvas = document.getElementById('canvasvideo');
const inputWidth = 513;
const inputHeight = 513;
const inputSize = [1, inputWidth, inputHeight, 3];
const videoWidth = 500;
const videoHeight = 500;
const algorithm = gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']);
let isMultiple = guiState.algorithm;

inputElement.addEventListener('change', () => {
  drawResult();
})

model.onFinishChange((model) => {
  guiState.model = model;
  main(currentTab === 'camera');
});

outputStride.onFinishChange((outputStride) => {
  guiState.outputStride = parseInt(outputStride);
  main(currentTab === 'camera');
});

scaleFactor.onFinishChange((scaleFactor) => {
  guiState.scaleFactor = parseFloat(scaleFactor);
  main(currentTab === 'camera');
});

useAtrousConv.onFinishChange((useAtrousConv) => {
  guiState.useAtrousConv = useAtrousConv;
  main(currentTab === 'camera');
});

scoreThreshold.onChange((scoreThreshold) => {
  guiState.scoreThreshold = parseFloat(scoreThreshold);
  utils._minScore = guiState.scoreThreshold;
  (currentTab === 'camera') ? poseDetectionFrame() : drawResult(false, false);
});

nmsRadius.onChange((nmsRadius) => {
  guiState.multiPoseDetection.nmsRadius = parseInt(nmsRadius);
  utils._nmsRadius = guiState.multiPoseDetection.nmsRadius;
  (currentTab === 'camera') ? poseDetectionFrame() : drawResult(false, true);
});

maxDetections.onChange((maxDetections) => {
  guiState.multiPoseDetection.maxDetections = parseInt(maxDetections);
  utils._maxDetection = guiState.multiPoseDetection.maxDetections;
  (currentTab === 'camera') ? poseDetectionFrame() : drawResult(false, true);
});

showPose.onChange((showPose) => {
  guiState.showPose = showPose;
  (currentTab === 'camera') ? poseDetectionFrame() : drawResult(false, false);
});

showBoundingBox.onChange((showBoundingBox) => {
  guiState.showBoundingBox = showBoundingBox;
  (currentTab === 'camera') ? poseDetectionFrame() : drawResult(false, false);
});

const drawImage = (image, canvas, w, h) => {
  const ctx = canvas.getContext('2d');
  canvas.width = w;
  canvas.height = h;
  canvas.setAttribute('width', w);
  canvas.setAttribute('height', h);
  ctx.save();
  ctx.drawImage(image, 0, 0, w, h);
  ctx.restore();
}

const loadImage = (imagePath, canvas) => {
  const ctx = canvas.getContext('2d');
  const image = new Image();
  const promise = new Promise((resolve, reject) => {
    image.crossOrigin = '';
    image.onload = () => {
      canvas.width = inputWidth;
      canvas.height = inputHeight;
      canvas.setAttribute('width', inputWidth);
      canvas.setAttribute('height', inputHeight);
      ctx.drawImage(image, 0, 0, inputWidth, inputHeight);
      resolve(image);
    };
  });
  image.src = imagePath;
  return promise;
}

let singlePose, multiPoses;
const drawResult = async (predict = true, decode = true) => {
  streaming = false;
  if(track) {
    track.stop();
  }
  try {
    let _inputElement = inputElement.files[0];
    let imageUrl;
    if (_inputElement != undefined) {
      imageUrl = await getInput(_inputElement);
    } else {
      imageUrl = '../skeleton_detection/img/download.png';
    }

    await loadImage(imageUrl, canvassingle);
    let image = await loadImage(imageUrl, canvasmulti);
    drawImage(image, scaleImage, utils.scaleWidth, utils.scaleHeight);
    let predictTime = 0, decodeTime = 0;
    if (predict) {
      await utils.predict(scaleImage, 'single');
      const start = performance.now();
      await utils.predict(scaleImage, 'multi');
      predictTime = performance.now() - start;
    }
    if (decode) {
      singlePose = utils.decodePose('single');
      const start = performance.now();
      multiPoses = utils.decodePose('multi');
      decodeTime = performance.now() - start;
    }
    if (predict && decode) {
      const elapsed = predictTime + decodeTime;
      const inferenceTimeElement = document.getElementById('inferenceTime');
      inferenceTimeElement.innerHTML = `inference: <span class='ir'>${elapsed.toFixed(2)} ms</span> predicting: <span class='ir'>${predictTime.toFixed(2)} ms</span> decoding: <span class='ir'>${decodeTime.toFixed(2)} ms</span>`;
    }
    utils.drawPoses(canvassingle, singlePose);
    utils.drawPoses(canvasmulti, multiPoses);
    showResults();
  }
  catch (e) {
    errorHandler(e);
  }
}

const setupCamera = async () => {
  showProgress('Starting camera ...');
  const stream = await navigator.mediaDevices.getUserMedia({ 'audio': false, 'video': {facingMode: 'user'}});
  video.srcObject = stream;
  track = stream.getTracks()[0];
  streaming = true;
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    }
  });
}

const loadVideo = async () => {
  const videoElement = await setupCamera();
  videoElement.play();
  canvas.setAttribute('width', videoElement.videoWidth);
  canvas.setAttribute('height', videoElement.videoHeight);
  return videoElement;
}

const predict = async (video) => {
  stats.begin();
  const start = performance.now();
  let type = guiState.algorithm == 'multi-pose' ? 'multi' : 'single';
  drawVideo(video, scaleCanvas, utils.scaleWidth, utils.scaleHeight);
  await utils.predict(scaleCanvas, type);
  drawVideo(video, canvas, video.videoWidth, video.videoHeight);
  utils.drawPoses(canvas, utils.decodePose(type));
  const elapsed = performance.now() - start;
  const inferenceTimeElement = document.getElementById('inferenceTime');
  inferenceTimeElement.innerHTML = `inference: <span class='ir'>${elapsed.toFixed(2)} ms</span>`;
  stats.end();
}

const drawVideo = (video, canvas, w, h) => {
  const ctx = canvas.getContext('2d');
  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-w, 0);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();
}

const poseDetectionFrame = async () => {
  if (streaming) {
    if (utils.initialized) {
      await predict(video);
    }
    setTimeout(poseDetectionFrame, 0);
  }
  showResults();
}

const updateScenario = async (camera = false) => {
  console.log(`Backend: ${currentBackend}, Prefer: ${currentPrefer}`);
  streaming = false;
  try {
    if(camera){
      await loadVideo();
      showProgress('Inferencing ...');
      poseDetectionFrame();
    }
    else {
      showProgress('Inferencing ...');
      drawResult();
    }
  } catch (e) {
    errorHandler(e);
  }
}

const main = async (camera = false) => {
  console.log(`Backend: ${currentBackend}, Prefer: ${currentPrefer}`);
  streaming = false;
  try { utils.deleteAll(); } catch (e) {}
  try {
    if(camera){
      await loadVideo();
      showProgress('Loading model and initializing ...');
      getOffloadOps(currentBackend, currentPrefer);
      await utils.init(currentBackend, currentPrefer, inputSize);
      showSubGraphsSummary(utils.getSubgraphsSummary());
      showProgress('Inferencing ...');
      poseDetectionFrame();
    }
    else {
      showProgress('Loading model and initializing...');
      getOffloadOps(currentBackend, currentPrefer);
      await utils.init(currentBackend, currentPrefer, inputSize);
      showSubGraphsSummary(utils.getSubgraphsSummary());
      showProgress('Inferencing ...');
      drawResult();
    }
  } catch (e) {
    errorHandler(e);
  }
}