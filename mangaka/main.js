function main() {

  const inputCanvas = document.getElementById('inputCanvas');
  const outputCanvas = document.getElementById('outputCanvas');
  const inputFile = document.getElementById('input');
  const inputButton = document.getElementById('button');
  const color = document.getElementById('color');
  const sketch = document.getElementById('sketch');
  const simplificaiton = document.getElementById('simplificaiton');
  const target = document.getElementById('target');
  const inputHeight = 256;
  const inputWidth = 256;
  let model;
  let currentModel = 'colorModel';

  init(currentModel);
  
  async function init(currentModel) {
    inputFile.setAttribute('disabled', "disabled");
    model = await tf.loadModel('./model/'+currentModel+'/model.json');
    model.summary();
    await start('./images/5.jpg');
    inputFile.removeAttribute('disabled');
    inputButton.innerHTML = "上传";
    inputButton.appendChild(inputFile);
    inputButton.setAttribute('class', 'btn btn-primary fileinput-button');
  }

  inputFile.addEventListener('change', (e) => {
    let files = e.target.files;
    if (files.length > 0) {
      let imagePath = URL.createObjectURL(files[0]);
      start(imagePath);
    }
  }, false);

  color.addEventListener('click', (e) => {
    clearCanvas();
    init('colorModel');
    target.innerHTML = '线稿上色图片';
    sketch.removeAttribute('class', 'active');
    simplificaiton.removeAttribute('class', 'active');
    color.setAttribute('class', 'active');
  }, false);

  sketch.addEventListener('click', (e) => {
    clearCanvas();
    init('colorModel');
    target.innerHTML = '线稿生成图片';
    color.removeAttribute('class', 'active');
    simplificaiton.removeAttribute('class', 'active');
    sketch.setAttribute('class', 'active');
  }, false);

  simplificaiton.addEventListener('click', (e) => {
    clearCanvas();
    init('colorModel');
    target.innerHTML = '线稿简化图片';
    color.removeAttribute('class','active');
    sketch.removeAttribute('class', 'active');
    simplificaiton.setAttribute('class', 'active');
  }, false);

  async function start(imagePath) {
    await loadImage(imagePath, inputCanvas);

    let outputTensor = tf.tidy(() => {
      let inputTensor = tf.fromPixels(inputCanvas); // int32, [256, 256, 3]
      return compute(inputTensor);
    })

    tf.toPixels(outputTensor, outputCanvas);
  }

  function loadImage(imagePath, canvas) {
    const ctx = canvas.getContext('2d');
    const image = new Image();
    const promise = new Promise((resolve, reject) => {
      image.onload = () => {
        ctx.drawImage(image, 0, 0, inputWidth, inputHeight);
        resolve(image);
      };
    });
    image.src = imagePath;
    return promise;
  }  

  function compute(inputTensor) {
    // preprocess: int32, [0, 255], [256, 256, 3]    =>    float, [-1, 1], [1, 256, 256, 3]
    let offset = tf.scalar(127.5);
    let preprocessedInputTensor = inputTensor.toFloat().div(offset).sub(tf.scalar(1.0)).expandDims();

    // predict: [-1, 1] => [-1, 1]
    let outputTensor = model.predict(preprocessedInputTensor);

    // deprocess: float, [-1, 1], [1, 256, 256, 3]   =>   float, [0, 1], [256, 256, 3]
    let scalar = tf.scalar(0.5);
    let deprocessedOutputTensor = outputTensor.mul(scalar).add(scalar).squeeze();

    return deprocessedOutputTensor;
  }

  function clearCanvas() {
    const inputCanvasContext = inputCanvas.getContext('2d');
    const outputCanvasContext = outputCanvas.getContext('2d');
    inputCanvasContext.clearRect(0, 0, 256, 256);
    outputCanvasContext.clearRect(0, 0, 256, 256);
  }
} 